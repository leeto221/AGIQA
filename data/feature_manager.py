# data/feature_manager.py
import os
import sys
import importlib
import numpy as np
import torch
import torch.nn.functional as F


class FeatureManager:
    """
    保留工程化缓存管理，但输出后处理对齐项目一：
    - feat: layer_norm
    - map: resize 到 im_hw + min-max normalize
    - task_type: quality / alignment
    - reiqa_mode: quality / content
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.task_type = config.task_type  # quality / alignment
        self.reiqa_mode = "quality" if self.task_type == "quality" else "content"

        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.reiqa_abs_path = os.path.join(self.project_root, config.reiqa_root)
        self.reiqa_parent = os.path.dirname(self.reiqa_abs_path)
        self.reiqa_pkg_name = os.path.basename(self.reiqa_abs_path)

        if self.reiqa_parent not in sys.path:
            sys.path.insert(0, self.reiqa_parent)
        if self.reiqa_abs_path not in sys.path:
            sys.path.insert(0, self.reiqa_abs_path)

        self.save_dir = os.path.join(
            self.project_root,
            config.cache_root,
            config.dataset_name,
            self.reiqa_mode
        )

        ckpt_name = "quality_aware_r50.pth" if self.reiqa_mode == "quality" else "content_aware_r50.pth"
        self.ckpt = os.path.join(self.reiqa_abs_path, "reiqa_ckpts", ckpt_name)

        self.feat_model = None
        self.map_model = None

    def _init_models(self):
        if self.feat_model is None or self.map_model is None:
            try:
                from inference_feats import ReIQAFeatExtractor
                from inference_maps import ReIQAMapExtractor
            except ImportError:
                ReIQAFeatExtractor = importlib.import_module(
                    f"{self.reiqa_pkg_name}.inference_feats"
                ).ReIQAFeatExtractor
                ReIQAMapExtractor = importlib.import_module(
                    f"{self.reiqa_pkg_name}.inference_maps"
                ).ReIQAMapExtractor

            print(f"--- [Loading ReIQA {self.reiqa_mode.upper()} model to {self.device}] ---")
            self.feat_model = ReIQAFeatExtractor(self.ckpt, str(self.device))
            self.map_model = ReIQAMapExtractor(self.ckpt, str(self.device))

    def _feat_cache_name(self, feat_stem):
        return f"{feat_stem}_{self.reiqa_mode}_aware_features.npy"

    def _map_cache_name(self, feat_stem):
        return f"{feat_stem}_{self.reiqa_mode}_map.npy"

    def _postprocess_feat(self, feat_np):
        feat = torch.from_numpy(feat_np).float()
        feat = feat.squeeze(0)
        feat = F.layer_norm(feat, feat.shape[-1:])
        return feat

    def _postprocess_map(self, map_np, target_hw):
        m = torch.from_numpy(map_np).float()

        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(0)
        elif m.dim() != 4:
            raise ValueError(f"Unsupported map shape: {m.shape}")

        m = F.interpolate(m, size=target_hw, mode='bilinear', align_corners=False)

        minv = m.amin(dim=[2, 3], keepdim=True)
        maxv = m.amax(dim=[2, 3], keepdim=True)
        m = (m - minv) / (maxv - minv + 1e-6)

        return m

    def get_data(self, img_path, feat_stem, is_training=True):
        feat_cache_path = os.path.join(self.save_dir, self._feat_cache_name(feat_stem))
        map_cache_path = os.path.join(self.save_dir, self._map_cache_name(feat_stem))

        if is_training and os.path.exists(feat_cache_path) and os.path.exists(map_cache_path):
            feat_np = np.load(feat_cache_path, allow_pickle=False)
            map_np = np.load(map_cache_path, allow_pickle=False)
            return feat_np, map_np

        os.makedirs(self.save_dir, exist_ok=True)
        self._init_models()

        feat_np = self.feat_model.extract(img_path, half_scale=True)
        map_np = self.map_model.extract_map(img_path, mode=self.reiqa_mode)

        if is_training:
            np.save(feat_cache_path, feat_np)
            np.save(map_cache_path, map_np)

        return feat_np, map_np

    def get_batch_data(self, img_paths, feat_stems, im_hws, is_training=True):
        batch_feats = []
        batch_maps = []

        for img_p, stem, hw in zip(img_paths, feat_stems, im_hws):
            feat_np, map_np = self.get_data(img_p, stem, is_training=is_training)

            feat = self._postprocess_feat(feat_np)

            H = int(hw[0])
            W = int(hw[1])
            qmap = self._postprocess_map(map_np, target_hw=(H, W)).squeeze(0)  # [1,H,W]

            batch_feats.append(feat)
            batch_maps.append(qmap)

        aux_batch = torch.stack(batch_feats, dim=0).to(self.device)  # [B,4096]
        qmaps = torch.stack(batch_maps, dim=0).to(self.device)       # [B,1,H,W]

        return aux_batch, qmaps
