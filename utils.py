# utils.py
import yaml
import torch
from scipy.stats import spearmanr, pearsonr

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BILINEAR = Image.BILINEAR


class Config:
    def __init__(self, entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Config(v)
            else:
                self.__dict__[k] = v


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config(data)


def compute_metrics(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    srcc, _ = spearmanr(y_pred, y_true)
    plcc, _ = pearsonr(y_pred, y_true)
    return float(srcc), float(plcc)


def loss_m3(y_pred, y, epoch=0):
    return torch.mean(torch.abs(y_pred - y))


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


class AdaptiveResize(object):
    def __init__(self, size, interpolation=BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        self.image_size = image_size

    def __call__(self, img):
        h, w = img.size
        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)
        if h < self.size or w < self.size:
            return transforms.Resize(self.size, self.interpolation)(img)
        else:
            return img


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_preprocess_train():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(512),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        ),
    ])


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
