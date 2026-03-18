# ReIQA-main/inference_feats.py
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
# 保持原有的相对导入，确保在 ReIQA-main 目录下运行正常
from options.train_options import TrainOptions
from networks.build_backbone import build_model

class ReIQAFeatExtractor:
    def __init__(self, ckpt_path, device, head="mlp"):
        self.device = torch.device(device)
        # 1. 模拟命令行参数初始化模型
        import sys
        # 暂时清空 argv 防止 TrainOptions 解析到外部训练脚本的参数
        _old_argv = sys.argv
        sys.argv = [sys.argv[0]] 
        
        args = TrainOptions().parse()
        args.head = head
        model, _ = build_model(args)
        model = torch.nn.DataParallel(model)

        # 2. 加载权重
        state = torch.load(ckpt_path, map_location="cpu")
        payload = state["model"] if (isinstance(state, dict) and "model" in state) else state
        model.load_state_dict(payload, strict=True)
        
        self.model = model.to(self.device).eval()
        sys.argv = _old_argv # 还原参数
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, img_path, half_scale=True):
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        
        # 原始尺度
        x1 = self.preprocess(img).unsqueeze(0).to(self.device)
        f1 = self.model.module.encoder(x1)
        
        if half_scale:
            # 半尺度
            img_half = img.resize((max(1, img.width // 2), max(1, img.height // 2)))
            x2 = self.preprocess(img_half).unsqueeze(0).to(self.device)
            f2 = self.model.module.encoder(x2)
            feat = torch.cat([f1, f2], dim=1)
        else:
            feat = f1
            
        return feat.detach().cpu().numpy() # [1, Dim]
