from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import random


class SRDataset(Dataset):
    def __init__(self, LR_path, GT_path, in_memory=True, transform=None):
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.in_memory = in_memory
        self.transform = transform

        self.LR_files = sorted(os.listdir(LR_path))
        self.GT_files = sorted(os.listdir(GT_path))

        if in_memory:
            self.LR_imgs = [
                np.array(Image.open(os.path.join(self.LR_path, f)).convert("RGB")).astype(np.uint8)
                for f in self.LR_files
            ]
            self.GT_imgs = [
                np.array(Image.open(os.path.join(self.GT_path, f)).convert("RGB")).astype(np.uint8)
                for f in self.GT_files
            ]

    def __len__(self):
        return len(self.LR_files)

    def __getitem__(self, i):
        if self.in_memory:
            GT = self.GT_imgs[i].astype(np.float32)
            LR = self.LR_imgs[i].astype(np.float32)
        else:
            GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_files[i])).convert("RGB"))
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_files[i])).convert("RGB"))

        img_item = {
            'GT': (GT / 127.5) - 1.0,
            'LR': (LR / 127.5) - 1.0,
        }

        if self.transform is not None:
            img_item = self.transform(img_item)

        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32)
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)

        return img_item


class TestOnlyDataset(Dataset):
    def __init__(self, LR_path, in_memory=True):
        self.LR_path = LR_path
        self.LR_files = sorted(os.listdir(LR_path))
        self.in_memory = in_memory

        if in_memory:
            self.LR_imgs = [
                np.array(Image.open(os.path.join(self.LR_path, f)))
                for f in self.LR_files
            ]

    def __len__(self):
        return len(self.LR_files)

    def __getitem__(self, i):
        if self.in_memory:
            LR = self.LR_imgs[i]
        else:
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_files[i])))

        LR = (LR / 127.5) - 1.0
        LR = LR.transpose(2, 0, 1).astype(np.float32)

        return {'LR': LR}


class RandomCrop:
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size

    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']
        ih, iw = LR_img.shape[:2]

        ix = random.randrange(0, iw - self.patch_size + 1)
        iy = random.randrange(0, ih - self.patch_size + 1)

        tx = ix * self.scale
        ty = iy * self.scale

        LR_patch = LR_img[iy:iy + self.patch_size, ix:ix + self.patch_size]
        GT_patch = GT_img[ty:ty + self.scale * self.patch_size, tx:tx + self.scale * self.patch_size]

        return {'LR': LR_patch, 'GT': GT_patch}


class RandomAugment:
    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']

        if random.randrange(0, 2):
            LR_img = np.fliplr(LR_img).copy()
            GT_img = np.fliplr(GT_img).copy()

        if random.randrange(0, 2):
            LR_img = np.flipud(LR_img).copy()
            GT_img = np.flipud(GT_img).copy()

        if random.randrange(0, 2):
            LR_img = LR_img.transpose(1, 0, 2)
            GT_img = GT_img.transpose(1, 0, 2)

        return {'LR': LR_img, 'GT': GT_img}
