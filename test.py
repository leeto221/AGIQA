# test.py
import os
import sys
import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms

from models.main_arch import DPGFNet, qsel_from_map


try:
    from torchvision.transforms import InterpolationMode
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BILINEAR = Image.BILINEAR


class AdaptiveResize(object):
    """
    与训练/验证阶段保持一致
    """
    def __init__(self, size, interpolation=BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        self.image_size = image_size

    def __call__(self, img):
        h, w = img.size  # 保持和原工程一致的写法
        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)
        if h < self.size or w < self.size:
            return transforms.Resize(self.size, self.interpolation)(img)
        return img


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_preprocess_val():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(512),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        ),
    ])


def build_patches_for_test(image_path, preprocess, num_patch=15):
    """
    对齐验证集逻辑：
    - preprocess_val
    - unfold 切 patch
    - 均匀采样 num_patch 个 patch
    - 拼 whole-image resized patch
    """
    img = Image.open(image_path).convert("RGB")
    I = preprocess(img)          # [3,H,W]
    I = I.unsqueeze(0)           # [1,3,H,W]

    n_channels = 3
    kernel_h = 224
    kernel_w = 224
    step = 48 if (I.size(2) >= 1024) or (I.size(3) >= 1024) else 32

    patches = (
        I.unfold(2, kernel_h, step)
         .unfold(3, kernel_w, step)
         .permute(2, 3, 0, 1, 4, 5)
         .reshape(-1, n_channels, kernel_h, kernel_w)
    )

    if patches.size(0) < num_patch:
        raise ValueError(
            f"patch count {patches.size(0)} < num_patch {num_patch}, "
            f"image={image_path}, size={tuple(I.shape)}"
        )

    sel_step = patches.size(0) // num_patch
    sel = torch.zeros(num_patch, dtype=torch.long)
    for i in range(num_patch):
        sel[i] = sel_step * i

    patches = patches[sel, ...]

    I_resized = F.interpolate(
        I, size=(kernel_h, kernel_w), mode='bilinear', align_corners=False
    )
    patches = torch.cat([patches, I_resized], dim=0)  # [Np+1,3,224,224]

    meta = {
        "sel": sel,
        "step": torch.tensor(step, dtype=torch.int64),
        "im_hw": torch.tensor([I.size(2), I.size(3)], dtype=torch.int64),
        "kernel_hw": torch.tensor([kernel_h, kernel_w], dtype=torch.int64),
    }

    return patches.unsqueeze(0), meta  # [1,Np+1,3,224,224]


def load_reiqa_extractors(reiqa_root, device, task_type):
    """
    动态加载 ReIQA 的特征提取器和 map 提取器
    """
    reiqa_abs_path = os.path.abspath(reiqa_root)
    reiqa_parent = os.path.dirname(reiqa_abs_path)
    reiqa_pkg_name = os.path.basename(reiqa_abs_path)

    if reiqa_parent not in sys.path:
        sys.path.insert(0, reiqa_parent)
    if reiqa_abs_path not in sys.path:
        sys.path.insert(0, reiqa_abs_path)

    try:
        from inference_feats import ReIQAFeatExtractor
        from inference_maps import ReIQAMapExtractor
    except ImportError:
        ReIQAFeatExtractor = importlib.import_module(
            f"{reiqa_pkg_name}.inference_feats"
        ).ReIQAFeatExtractor
        ReIQAMapExtractor = importlib.import_module(
            f"{reiqa_pkg_name}.inference_maps"
        ).ReIQAMapExtractor

    reiqa_mode = "quality" if task_type == "quality" else "content"
    ckpt_name = "quality_aware_r50.pth" if reiqa_mode == "quality" else "content_aware_r50.pth"
    reiqa_ckpt = os.path.join(reiqa_abs_path, "reiqa_ckpts", ckpt_name)

    feat_extractor = ReIQAFeatExtractor(reiqa_ckpt, str(device))
    map_extractor = ReIQAMapExtractor(reiqa_ckpt, str(device))

    return feat_extractor, map_extractor, reiqa_mode


def postprocess_feat(feat_np, device):
    feat = torch.from_numpy(feat_np).float().squeeze(0)
    feat = F.layer_norm(feat, feat.shape[-1:])
    return feat.unsqueeze(0).to(device)  # [1,4096]


def postprocess_map(map_np, target_hw, device):
    m = torch.from_numpy(map_np).float()
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:
        m = m.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported map shape: {m.shape}")

    m = F.interpolate(m, size=target_hw, mode='bilinear', align_corners=False)
    minv = m.amin(dim=[2, 3], keepdim=True)
    maxv = m.amax(dim=[2, 3], keepdim=True)
    m = (m - minv) / (maxv - minv + 1e-6)
    return m.to(device)  # [1,1,H,W]


def maybe_save_cache(cache_root, dataset_name, task_type, image_path, feat_np, map_np):
    """
    你说测试时不在乎命中旧缓存，但新生成后顺手存一下通常更方便。
    不想存可以把这个调用删掉。
    """
    if cache_root is None:
        return

    reiqa_mode = "quality" if task_type == "quality" else "content"
    stem = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(cache_root, dataset_name, reiqa_mode)
    os.makedirs(save_dir, exist_ok=True)

    feat_path = os.path.join(save_dir, f"{stem}_{reiqa_mode}_aware_features.npy")
    map_path = os.path.join(save_dir, f"{stem}_{reiqa_mode}_map.npy")

    np.save(feat_path, feat_np)
    np.save(map_path, map_np)


def infer_one_image(
    ckpt_path,
    image_path,
    prompt,
    task_type,
    device="cuda:0",
    reiqa_root="./ReIQA_main",
    cache_root=None,
    dataset_name="single_test",
    clip_model_name="ViT-B/32",
    val_num_patch=15,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1) 加载模型
    model = DPGFNet(
        clip_model_name=clip_model_name,
        device=device
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 2) 图像 -> patch
    preprocess = get_preprocess_val()
    patches, meta = build_patches_for_test(
        image_path=image_path,
        preprocess=preprocess,
        num_patch=val_num_patch
    )
    patches = patches.to(device)

    # 3) 在线生成 ReIQA feature 和 map
    feat_extractor, map_extractor, reiqa_mode = load_reiqa_extractors(
        reiqa_root=reiqa_root,
        device=device,
        task_type=task_type
    )

    feat_np = feat_extractor.extract(image_path, half_scale=True)
    map_np = map_extractor.extract_map(image_path, mode=reiqa_mode)

    # 可选缓存保存
    maybe_save_cache(
        cache_root=cache_root,
        dataset_name=dataset_name,
        task_type=task_type,
        image_path=image_path,
        feat_np=feat_np,
        map_np=map_np
    )

    # 4) 后处理，严格对齐训练/验证
    aux_vec = postprocess_feat(feat_np, device=device)  # [1,4096]
    qmap = postprocess_map(
        map_np,
        target_hw=(int(meta["im_hw"][0]), int(meta["im_hw"][1])),
        device=device
    )  # [1,1,H,W]

    # 5) 从 map 提取 qsel
    qsel = qsel_from_map(
        qmap[0],
        (int(meta["kernel_hw"][0]), int(meta["kernel_hw"][1])),
        int(meta["step"]),
        meta["sel"]
    )
    qsel_batch = [qsel.detach()]

    # 6) 前向
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        preds = model(
            x=patches,
            prompt=[prompt],
            qsel_batch=qsel_batch,
            aux_batch=aux_vec,
            freeze_prior=False
        )

    return float(preds.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--image", type=str, required=True, help="待测试图片路径")
    parser.add_argument("--prompt", type=str, required=True, help="该图像对应的文本 prompt")
    parser.add_argument("--task_type", type=str, required=True, choices=["quality", "alignment"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--reiqa_root", type=str, default="./ReIQA_main")
    parser.add_argument("--cache_root", type=str, default=None, help="可选；若提供则把新生成的 npy 存下来")
    parser.add_argument("--dataset_name", type=str, default="single_test", help="仅用于缓存目录命名")
    parser.add_argument("--clip_model_name", type=str, default="ViT-B/32")
    parser.add_argument("--val_num_patch", type=int, default=15)
    args = parser.parse_args()

    score = infer_one_image(
        ckpt_path=args.ckpt,
        image_path=args.image,
        prompt=args.prompt,
        task_type=args.task_type,
        device=args.device,
        reiqa_root=args.reiqa_root,
        cache_root=args.cache_root,
        dataset_name=args.dataset_name,
        clip_model_name=args.clip_model_name,
        val_num_patch=args.val_num_patch,
    )

    print("========================================")
    print(f"Image     : {args.image}")
    print(f"Task      : {args.task_type}")
    print(f"Prompt    : {args.prompt}")
    print(f"Score     : {score:.4f}")
    print("========================================")


if __name__ == "__main__":
    main()
