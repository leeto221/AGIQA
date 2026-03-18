# train.py
import os
import random
import numpy as np
import torch
import clip
import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from utils import (
    load_config,
    compute_metrics,
    get_preprocess_train,
    get_preprocess_val,
    convert_models_to_fp32,
    loss_m3,
)

from dataset.dataset_aigc import (
    AIGCDataset_3k,
    AIGCIQA2023Dataset,
    PKUI2IDataset,
)

from data.feature_manager import FeatureManager
from models.main_arch import DPGFNet, qsel_from_map


def set_seed(seed=20200626):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_model(model, opt=0):

    model.clip_model.logit_scale.requires_grad = False

    if opt == 0:
        return
    elif opt == 1:
        for p in model.clip_model.token_embedding.parameters():
            p.requires_grad = False
        for p in model.clip_model.transformer.parameters():
            p.requires_grad = False
        model.clip_model.positional_embedding.requires_grad = False
        model.clip_model.text_projection.requires_grad = False
        for p in model.clip_model.ln_final.parameters():
            p.requires_grad = False
    elif opt == 2:
        for p in model.clip_model.visual.parameters():
            p.requires_grad = False
    elif opt == 3:
        for p in model.clip_model.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"Unsupported freeze_opt: {opt}")


def build_datasets_and_loaders(cfg):
    dataset_map = {
        "AGIQA3k": AIGCDataset_3k,
        "AIGCIQA2023": AIGCIQA2023Dataset,
        "PKUI2IQA": PKUI2IDataset,
    }

    if cfg.dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset_name: {cfg.dataset_name}")

    DatasetClass = dataset_map[cfg.dataset_name]

    train_preprocess = get_preprocess_train()
    val_preprocess = get_preprocess_val()

    train_ds = DatasetClass(
        csv_file=cfg.train_csv,
        img_dir=cfg.img_dir,
        preprocess=train_preprocess,
        num_patch=getattr(cfg, "train_num_patch", 8),
        test=False,
        task_type=cfg.task_type,
    )

    val_ds = DatasetClass(
        csv_file=cfg.val_csv,
        img_dir=cfg.img_dir,
        preprocess=val_preprocess,
        num_patch=getattr(cfg, "val_num_patch", 15),
        test=True,
        task_type=cfg.task_type,
    )

    batch_size = getattr(cfg, "batch_size", 16)
    num_workers = getattr(cfg, "num_workers", 16)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def extract_qsel_batch(batch, qmaps, device):
    qsel_batch = []
    for b in range(qmaps.size(0)):
        qmap_b = qmaps[b].to(device)  # [1,H,W]
        kh = int(batch['kernel_hw'][b, 0])
        kw = int(batch['kernel_hw'][b, 1])
        stp = int(batch['step'][b])
        sel = batch['sel'][b]

        qsel = qsel_from_map(qmap_b, (kh, kw), stp, sel)
        qsel_batch.append(qsel.detach())
    return qsel_batch


def build_optimizer(cfg, model):
    clip_lr = getattr(cfg, "clip_lr", 5e-6)
    tcpga_lr = getattr(cfg, "tcpga_lr", 5e-5)
    prior_gate_lr = getattr(cfg, "prior_gate_lr", 1e-5)
    full_mod_lr = getattr(cfg, "full_mod_lr", 5e-5)

    clip_wd = getattr(cfg, "clip_weight_decay", 0.001)
    others_wd = getattr(cfg, "others_weight_decay", 0.0)

    optimizer = torch.optim.AdamW([
        {'params': model.clip_model.parameters(), 'lr': clip_lr, 'weight_decay': clip_wd},
        {'params': model.tcpga.parameters(), 'lr': tcpga_lr, 'weight_decay': others_wd},
        {'params': model.prior_gate.parameters(), 'lr': prior_gate_lr, 'weight_decay': others_wd},
        {'params': model.full_mod.parameters(), 'lr': full_mod_lr, 'weight_decay': others_wd},
    ])
    return optimizer


def build_scheduler(cfg, optimizer):

    scheduler_mode = getattr(cfg, "scheduler_mode", "cosine")
    clip_lr = getattr(cfg, "clip_lr", 5e-6)

    if scheduler_mode == "cosine":
        t_max = getattr(cfg, "cosine_t_max", 5)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return {"mode": "cosine", "scheduler": scheduler}

    elif scheduler_mode == "warmup_cosine":
        warmup_epochs = getattr(cfg, "warmup_epochs", 3)
        epochs = getattr(cfg, "epochs", 25)
        eta_min_factor = getattr(cfg, "cosine_eta_min_factor", 0.2)

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=clip_lr * eta_min_factor
        )
        return {
            "mode": "warmup_cosine",
            "scheduler": scheduler,
            "warmup_epochs": warmup_epochs
        }

    else:
        raise ValueError(f"Unsupported scheduler_mode: {scheduler_mode}")


def adjust_learning_rate_with_warmup(optimizer, epoch, warmup_epochs):
    if epoch < warmup_epochs:
        lr_scale = (epoch + 1) / warmup_epochs
    else:
        lr_scale = 1.0

    for param_group in optimizer.param_groups:
        if 'base_lr' not in param_group:
            param_group['base_lr'] = param_group['lr']
        param_group['lr'] = param_group['base_lr'] * lr_scale


def train_one_epoch(model, train_loader, feature_manager, optimizer, device, epoch):
    model.eval()  

    running_loss = 0.0
    freeze_prior = (epoch < 3)

    loop = tqdm.tqdm(train_loader, desc=f"Epoch:{epoch}")

    for step, batch in enumerate(loop):
        patches = batch['I'].to(device)
        mos = batch['mos'].to(device).float()
        prompts = batch['prompt']

        aux_vec, qmaps = feature_manager.get_batch_data(
            img_paths=batch['image_path'],
            feat_stems=batch['feat_stem'],
            im_hws=batch['im_hw'],
            is_training=True,
        )

        qsel_batch = extract_qsel_batch(batch, qmaps, device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            preds = model(
                patches,
                prompts,
                qsel_batch,
                aux_vec,
                freeze_prior=freeze_prior
            )
            loss = loss_m3(preds, mos)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at step {step}, skipping...")
            optimizer.zero_grad()
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.tcpga.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(model.prior_gate.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(model.full_mod.parameters(), 0.5)

        if device.type == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model.clip_model)

        running_loss += float(loss.detach())
        avg_loss = running_loss / (step + 1)
        loop.set_description(f"Epoch:{epoch} Loss:{avg_loss:.4f}")

    return running_loss / max(len(train_loader), 1)


@torch.no_grad()
def validate(model, val_loader, feature_manager, device):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        for batch in tqdm.tqdm(val_loader, desc="Validation"):
            patches = batch['I'].to(device)
            mos = batch['mos'].to(device).float()
            prompts = batch['prompt']

            aux_vec, qmaps = feature_manager.get_batch_data(
                img_paths=batch['image_path'],
                feat_stems=batch['feat_stem'],
                im_hws=batch['im_hw'],
                is_training=True,
            )

            qsel_batch = extract_qsel_batch(batch, qmaps, device)

            preds = model(
                patches,
                prompts,
                qsel_batch,
                aux_vec,
                freeze_prior=False
            )

            valid_mask = ~torch.isnan(preds) & ~torch.isinf(preds)
            if valid_mask.sum() < len(preds):
                print(f"Warning: filtered {len(preds) - int(valid_mask.sum())} invalid predictions")

            all_preds.append(preds[valid_mask])
            all_targets.append(mos[valid_mask])

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    srcc, plcc = compute_metrics(all_preds, all_targets)
    score = (srcc + plcc) / 2.0

    return score, srcc, plcc


def main(config_path):
    cfg = load_config(config_path)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    set_seed(getattr(cfg, "seed", 20200626))

    train_loader, val_loader = build_datasets_and_loaders(cfg)

    model = DPGFNet(
        clip_model_name=getattr(cfg, "clip_model_name", "ViT-B/32"),
        device=device
    ).to(device)

    feature_manager = FeatureManager(cfg)

    optimizer = build_optimizer(cfg, model)
    scheduler_bundle = build_scheduler(cfg, optimizer)

    freeze_model(model, opt=getattr(cfg, "freeze_opt", 0))

    epochs = getattr(cfg, "epochs", 45)
    checkpoint_dir = getattr(cfg, "checkpoint_dir", os.path.join("models", "checkpoint"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_score = -1.0
    best_srcc = -1.0
    best_plcc = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f"Current LR: {current_lr}")

        if scheduler_bundle["mode"] == "warmup_cosine":
            adjust_learning_rate_with_warmup(
                optimizer,
                epoch,
                scheduler_bundle["warmup_epochs"]
            )

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            feature_manager=feature_manager,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        score, srcc, plcc = validate(
            model=model,
            val_loader=val_loader,
            feature_manager=feature_manager,
            device=device
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"score={score:.4f}, srcc={srcc:.4f}, plcc={plcc:.4f}"
        )

        if score > best_score:
            best_score = score
            best_srcc = srcc
            best_plcc = plcc
            best_epoch = epoch

            save_path = os.path.join(checkpoint_dir, f"{cfg.experiment_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")

        if scheduler_bundle["mode"] == "cosine":
            scheduler_bundle["scheduler"].step()
        elif scheduler_bundle["mode"] == "warmup_cosine":
            if epoch >= scheduler_bundle["warmup_epochs"]:
                scheduler_bundle["scheduler"].step()

    print("Training finished.")
    print(
        f"Best epoch={best_epoch}, "
        f"best score={best_score:.4f}, "
        f"best srcc={best_srcc:.4f}, "
        f"best plcc={best_plcc:.4f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    main(args.config)
