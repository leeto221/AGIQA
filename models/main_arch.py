# models/main_arch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from itertools import product


def qsel_from_map(qmap, kernel_hw, step, sel):
    """
    qmap: [1,1,H,W] or [1,H,W] or [H,W]
    kernel_hw: (kh, kw)
    sel: [Np]
    return: [Np]
    """
    if qmap.dim() == 2:
        qmap = qmap.unsqueeze(0).unsqueeze(0)
    elif qmap.dim() == 3:
        qmap = qmap.unsqueeze(0)

    kh, kw = kernel_hw
    uf = F.unfold(qmap, kernel_size=(kh, kw), stride=int(step))
    q_all = uf.mean(dim=1).squeeze(0)
    return q_all[sel.long()]


class TCPGA(nn.Module):
    """Text-Conditioned Prior-Guided Aggregation"""
    def __init__(self, dim=512, d=256):
        super().__init__()
        self.d = d
        self.scale = d ** -0.5

        self.Wq = nn.Linear(dim, d, bias=False)
        self.Wk = nn.Linear(dim, d, bias=False)
        self.Wv = nn.Linear(dim, d, bias=False)
        self.proj_out = nn.Linear(d, dim, bias=False)

        nn.init.xavier_uniform_(self.Wq.weight, gain=0.1)
        nn.init.xavier_uniform_(self.Wk.weight, gain=0.1)
        nn.init.xavier_uniform_(self.Wv.weight, gain=0.1)
        nn.init.xavier_uniform_(self.proj_out.weight, gain=0.1)

    def forward(self, E_patches, T_b5, P_batch):
        Q = self.Wq(T_b5)       # [B, 5, d]
        K = self.Wk(E_patches)  # [B, Np, d]
        V = self.Wv(E_patches)  # [B, Np, d]

        scores = torch.einsum('bkd,bnd->bkn', Q, K) * self.scale  # [B, 5, Np]

        for b in range(E_patches.size(0)):
            P_clamped = P_batch[b].clamp(min=0.1, max=0.9)
            logP = torch.log(P_clamped)
            scores[b] = scores[b] + logP.unsqueeze(0)

        A = torch.softmax(scores, dim=-1)           # [B, 5, Np]
        F_agg = torch.einsum('bkn,bnd->bkd', A, V)  # [B, 5, d]
        F_agg = self.proj_out(F_agg)                # [B, 5, 512]
        F_agg = F_agg / (F_agg.norm(dim=-1, keepdim=True) + 1e-8)

        return F_agg, A


class PriorGate(nn.Module):
    """P = sigmoid(lambda_q * q + lambda_c * c + lambda_0)"""
    def __init__(self):
        super().__init__()
        self.lambda_q = nn.Parameter(torch.zeros(1))
        self.lambda_c = nn.Parameter(torch.zeros(1))
        self.lambda_0 = nn.Parameter(torch.zeros(1))

    def forward(self, qsel_batch, csel_batch=None, freeze_prior=False):
        if freeze_prior:
            return [torch.ones_like(qsel) * 0.5 for qsel in qsel_batch]

        P_list = []
        for b, qsel in enumerate(qsel_batch):
            logit = self.lambda_q * qsel + self.lambda_0
            if csel_batch is not None:
                logit = logit + self.lambda_c * csel_batch[b]
            P = torch.sigmoid(logit).clamp(min=0.1, max=0.9)
            P_list.append(P)
        return P_list


class FullMod(nn.Module):
    """Half-scale modulation of global embedding"""
    def __init__(self, vdim=2048, dim=512):
        super().__init__()
        hid = 256

        self.scale_head = nn.Sequential(
            nn.Linear(vdim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, dim)
        )
        self.shift_head = nn.Sequential(
            nn.Linear(vdim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, dim)
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(vdim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, 1)
        )

        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].bias)
        nn.init.zeros_(self.shift_head[-1].weight)
        nn.init.zeros_(self.shift_head[-1].bias)
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.zeros_(self.alpha_head[-1].bias)

    def forward(self, E_full, v_half):
        scale = 1.0 + 0.1 * torch.tanh(self.scale_head(v_half))
        shift = 0.1 * self.shift_head(v_half)
        alpha = torch.sigmoid(self.alpha_head(v_half))

        E_full_mod = E_full * scale + shift
        E_full_mod = E_full_mod / (E_full_mod.norm(dim=-1, keepdim=True) + 1e-8)
        return E_full_mod, alpha


class DPGFNet(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.device = device
        self.quality_templates = ['badly', 'poorly', 'fairly', 'well', 'perfectly']

        self.clip_model, _ = clip.load(clip_model_name, device=device, jit=False)

        self.tcpga = TCPGA(dim=512, d=256)
        self.prior_gate = PriorGate()
        self.full_mod = FullMod(vdim=2048, dim=512)

    def encode_inputs(self, x, prompt):
        B = x.size(0)

        texts = [
            f"a photo that {c} matches '{p}'"
            for p, c in product(prompt, self.quality_templates)
        ]
        input_texts = torch.cat([clip.tokenize(t, truncate=True) for t in texts]).to(self.device)

        x_flat = x.view(-1, x.size(2), x.size(3), x.size(4))
        E_flat = self.clip_model.encode_image(x_flat)
        E_all = E_flat.view(B, -1, 512)

        T_flat = self.clip_model.encode_text(input_texts)
        T_all = T_flat.view(B, 5, -1)

        return E_all, T_all

    def forward(self, x, prompt, qsel_batch, aux_batch, freeze_prior=False):
        E_all, T_all = self.encode_inputs(x, prompt)

        E_patches = E_all[:, :-1, :]
        E_full = E_all[:, -1, :]

        P_batch = self.prior_gate(qsel_batch, freeze_prior=freeze_prior)

        F_patch, _ = self.tcpga(E_patches, T_all, P_batch)

        logit_scale = self.clip_model.logit_scale.exp().clamp(max=100)
        T_all_norm = T_all / (T_all.norm(dim=-1, keepdim=True) + 1e-8)

        logits5_patch = torch.einsum('bkd,bkd->bk', F_patch, T_all_norm) * logit_scale

        v_half = aux_batch[:, 2048:]
        E_full_mod, alpha = self.full_mod(E_full, v_half)

        logits5_full = torch.einsum('bd,bkd->bk', E_full_mod, T_all_norm) * logit_scale

        logits5_fused = alpha * logits5_patch + (1 - alpha) * logits5_full

        probs5 = F.softmax(logits5_fused, dim=-1)
        weights5 = torch.arange(1, 6, device=x.device).float().unsqueeze(0)

        preds = (probs5 * weights5).sum(-1)
        preds = ((preds - 1) / 4) * 5

        return preds
