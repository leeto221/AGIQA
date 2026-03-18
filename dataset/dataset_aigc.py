# dataset/dataset_aigc.py
import os
import functools
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
ImageFile.LOAD_TRUNCATED_IMAGES = True


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if not has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        raise ValueError(f"Unsupported image extension: {image_name}")
    I = Image.open(image_name)
    return I.convert('RGB')


def get_default_img_loader():
    return functools.partial(image_loader)


class AIGCDataset_3k(Dataset):
    def __init__(self, csv_file, img_dir, preprocess, num_patch, test,
                 task_type='quality', blind=False, get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print(f'{len(self.data.index)} csv data successfully loaded for 3K ({task_type})!')

        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        self.task_type = task_type
        self.blind = blind

    def __getitem__(self, index):
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.img_dir, image_name)

        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)

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

        grid_h = (I.size(2) - kernel_h) // step + 1
        grid_w = (I.size(3) - kernel_w) // step + 1

        assert patches.size(0) >= self.num_patch, \
            f"patch count {patches.size(0)} < num_patch {self.num_patch}, image={image_name}, size={tuple(I.shape)}"

        mos_idx = 5 if self.task_type == 'quality' else 7

        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
            mos = 0.0 if self.blind else float(self.data.iloc[index, mos_idx])
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch,)).long()
            mos = float(self.data.iloc[index, mos_idx])

        patches = patches[sel, ...]
        I_resized = F.interpolate(I, size=(kernel_h, kernel_w), mode='bilinear', align_corners=False)
        patches = torch.cat([patches, I_resized], dim=0)

        prompt = self.data.iloc[index, 1]
        model_name = image_name.split('_')[0]
        prompt_name = model_name + ' ' + prompt
        feat_stem = os.path.splitext(image_name)[0]

        sample = {
            'I': patches,
            'mos': mos,
            'prompt': prompt,
            'prompt_name': prompt_name,
            'image_name': image_name,
            'image_path': image_path,
            'feat_stem': feat_stem,
            'sel': sel,
            'step': torch.tensor(step, dtype=torch.int64),
            'grid_hw': torch.tensor([grid_h, grid_w], dtype=torch.int64),
            'im_hw': torch.tensor([I.size(2), I.size(3)], dtype=torch.int64),
            'kernel_hw': torch.tensor([kernel_h, kernel_w], dtype=torch.int64),
        }
        return sample

    def __len__(self):
        return len(self.data.index)


class AIGCIQA2023Dataset(Dataset):
    def __init__(self, csv_file, img_dir, preprocess, num_patch, test,
                 task_type='quality', blind=False, get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print(f'{len(self.data.index)} csv data successfully loaded for 2023 ({task_type})!')

        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        self.task_type = task_type
        self.blind = blind

    def __getitem__(self, index):
        # CSV:
        # col 0 = model
        # col 1 = image_name
        model = self.data.iloc[index, 0]
        image_name = self.data.iloc[index, 1]

        # 你的真实磁盘文件名规则：
        # "model,image_name"
        disk_image_name = f"{model}{image_name}"
        image_path = os.path.join(self.img_dir, disk_image_name)

        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)

        n_channels = 3
        kernel_h, kernel_w = 224, 224
        step = 48 if (I.size(2) >= 1024) or (I.size(3) >= 1024) else 32

        patches = (
            I.unfold(2, kernel_h, step)
             .unfold(3, kernel_w, step)
             .permute(2, 3, 0, 1, 4, 5)
             .reshape(-1, n_channels, kernel_h, kernel_w)
        )

        grid_h = (I.size(2) - kernel_h) // step + 1
        grid_w = (I.size(3) - kernel_w) // step + 1

        assert patches.size(0) >= self.num_patch, \
            f"patch count {patches.size(0)} < num_patch {self.num_patch}, image={image_path}, size={tuple(I.shape)}"

        # 与项目一一致：
        # quality -> col 2
        # alignment -> col 4
        mos_idx = 2 if self.task_type == 'quality' else 4

        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.arange(0, self.num_patch) * sel_step
            sel = sel.long()
            mos = 0.0 if self.blind else float(self.data.iloc[index, mos_idx])
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch,)).long()
            mos = float(self.data.iloc[index, mos_idx])

        patches = patches[sel, ...]
        I_resized = F.interpolate(I, size=(kernel_h, kernel_w), mode='bilinear', align_corners=False)
        patches = torch.cat([patches, I_resized], dim=0)

        prompt = self.data.iloc[index, 5]

        # 继续保持项目一的缓存命名规则
        image_stem = os.path.splitext(image_name)[0]
        feat_stem = f"{model}{image_stem}"

        model_name = image_name.split('_')[0]
        prompt_name = model_name + ' ' + prompt

        sample = {
            'I': patches,
            'mos': mos,
            'prompt': prompt,
            'prompt_name': prompt_name,
            'image_name': disk_image_name,   # 磁盘真实文件名
            'image_path': image_path,
            'feat_stem': feat_stem,
            'sel': sel,
            'step': torch.tensor(step, dtype=torch.int64),
            'grid_hw': torch.tensor([grid_h, grid_w], dtype=torch.int64),
            'im_hw': torch.tensor([I.size(2), I.size(3)], dtype=torch.int64),
            'kernel_hw': torch.tensor([kernel_h, kernel_w], dtype=torch.int64),
        }
        return sample

    def __len__(self):
        return len(self.data.index)




class PKUI2IDataset(Dataset):
    def __init__(self, csv_file, img_dir, preprocess, num_patch, test,
                 task_type='quality', blind=False, get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file, sep=',')
        print(f'{len(self.data.index)} csv data successfully loaded for PKU ({task_type})!')

        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        self.task_type = task_type
        self.blind = blind

    def __getitem__(self, index):
        image_name = self.data.iloc[index, 2]
        text_prompt = self.data.iloc[index, 1]
        image_path = os.path.join(self.img_dir, image_name)

        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)

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

        grid_h = (I.size(2) - kernel_h) // step + 1
        grid_w = (I.size(3) - kernel_w) // step + 1

        assert patches.size(0) >= self.num_patch, \
            f"patch count {patches.size(0)} < num_patch {self.num_patch}, image={image_name}, size={tuple(I.shape)}"

        mos_idx = 3 if self.task_type == 'quality' else 5

        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
            mos = 0.0 if self.blind else float(self.data.iloc[index, mos_idx])
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch,)).long()
            mos = float(self.data.iloc[index, mos_idx])

        patches = patches[sel, ...]
        I_resized = F.interpolate(I, size=(kernel_h, kernel_w), mode='bilinear', align_corners=False)
        patches = torch.cat([patches, I_resized], dim=0)

        model_name = image_name.split('_')[0]
        prompt_name = model_name + ' ' + text_prompt
        feat_stem = os.path.splitext(image_name)[0]

        sample = {
            'I': patches,
            'mos': mos,
            'prompt': text_prompt,
            'prompt_name': prompt_name,
            'image_name': image_name,
            'image_path': image_path,
            'feat_stem': feat_stem,
            'sel': sel,
            'step': torch.tensor(step, dtype=torch.int64),
            'grid_hw': torch.tensor([grid_h, grid_w], dtype=torch.int64),
            'im_hw': torch.tensor([I.size(2), I.size(3)], dtype=torch.int64),
            'kernel_hw': torch.tensor([kernel_h, kernel_w], dtype=torch.int64),
        }
        return sample

    def __len__(self):
        return len(self.data.index)
