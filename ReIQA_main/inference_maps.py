# ReIQA-main/inference_maps.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
try:
    from .inference_feats import ReIQAFeatExtractor
except ImportError:
    from inference_feats import ReIQAFeatExtractor


class ReIQAMapExtractor(ReIQAFeatExtractor):
    def __init__(self, ckpt_path, device, head="mlp"):
        super().__init__(ckpt_path, device, head)

    def _get_fmap(self, x):
        enc = self.model.module.encoder
        bb = getattr(enc, "backbone", enc)
        
        # 优先寻找 forward_spatial
        if hasattr(bb, "forward_spatial"):
            out = bb.forward_spatial(x, return_pooled=True)
            return out[1] if isinstance(out, (tuple, list)) else out
        
        # 否则查找 layer4
        target = getattr(bb, "layer4", None)
        if target is None and hasattr(bb, "layers"):
            target = bb.layers[3]
            
        blob = {}
        def _hook(_m, _in, out): blob["fmap"] = out.detach()
        h = target.register_forward_hook(_hook)
        try:
            _ = self.model.module.encoder(x)
            return blob["fmap"]
        finally:
            h.remove()

    @torch.no_grad()
    def extract_map(self, img_path, mode="quality"):
        """mode: 'quality' (L2 Norm) or 'content' (ReLU Mean)"""
        img = Image.open(img_path).convert("RGB")
        orig_size = (img.height, img.width)
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        
        fmap = self._get_fmap(x)
        
        if mode == "quality":
            # 质量图：L2 范数
            low_res_map = torch.norm(fmap, p=2, dim=1, keepdim=True)
        else:
            # 内容图：ReLU + 均值
            low_res_map = torch.relu(fmap).mean(dim=1, keepdim=True)

        # 插值回原图大小
        m = F.interpolate(low_res_map, size=orig_size, mode="bilinear", align_corners=False)
        
        # 归一化 [0, 1]
        m = m - m.amin(dim=[2,3], keepdim=True)
        m = m / (m.amax(dim=[2,3], keepdim=True) + 1e-6)
        
        return m.squeeze().cpu().numpy().astype("float32") # (H, W)
