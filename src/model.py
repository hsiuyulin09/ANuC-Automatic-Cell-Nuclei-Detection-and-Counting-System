import random
import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch_directml
from torch.utils.data import Dataset

class DiceLoss(nn.Module): 
        # Dice Loss = 1 -  2 × |A ∩ B| / (|A| + |B|)  # A 預測區域, B 實際區域
        # Dice Loss = 1 - (2 × intersection + smooth) / (union + smooth)
    def __init__(self, smooth=0.000001): # smooth=0.000001 避免分母為 0
        super(DiceLoss, self).__init__() # super().__init__()
        self.smooth = smooth

    def forward(self, forward_output, target):
        probs = torch.sigmoid(forward_output)

        probs = probs.view(-1)
        target = target.view(-1)

        intersection = (probs * target).sum()
        union = (probs + target).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        return dice_loss


class BCEDiceLoss(nn.Module):
    def  __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, forward_output, target):
        bce_loss = self.bce(forward_output, target)
        dice_loss = self.dice(forward_output, target)

        bce_weight = self.bce_weight
        dice_weight = 1 - self.bce_weight

        loss = bce_loss * bce_weight + dice_loss * dice_weight

        return loss
    

class H5Dataset(Dataset):
    def __init__(self, h5_path, mode="train", augment=False, seed=77): # mode = '(.h5 group)'
        self.h5_path = h5_path
        self.mode = mode
        self.augment = augment
        self.seed = seed
        self.current_epoch = 0  # epoch counter

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, index):  # for dataset[index]
        with h5py.File(self.h5_path, "r") as f:
            img = f[f"{self.mode}/images"][index]  # img.shape = (256, 256, 3) = (H, W, C)
            mask = f[f"{self.mode}/masks"][index]  # mask.shape = (256, 256) = (H, W)

        if self.augment and self.mode == "train":
            rng = random.Random(self.seed + index + self.current_epoch)  # 格式random.Random(seed)
            k = rng.randint(0, 3)  # 0到3隨機int
            img = np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
            mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))
                # np.ascontiguousarray整理記憶體位置 #np.rot90(target, 次數, 參與旋轉的維度)旋轉np矩陣一次90度 # axes=(0, 1)指img, mask .shape的(H, W, C), (H, W)對應索引位置(0, 1, 2), (0, 1)
            brightness_factor = rng.uniform(0.8, 1.2) # 隨機亮度調整
            img = img.astype(np.float32) * brightness_factor
            img = np.clip(img, 0, 255).astype(np.uint8)  # np.clip(target, min, max) 限制數據範圍超過max變max, 低於min變min

        img_t = torch.from_numpy(img[:, :, 0]).float().unsqueeze(0) / 255.0  # img.shape = (256, 256, 3) >>> img[:, :, 0] = (256, 256) >>> .unsqueeze(0) = (1, 256, 256) = (C, H, W)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        return img_t, mask_t

    def __len__(self):
        with h5py.File(self.h5_path, "r") as f:
            return len(f[f"{self.mode}/images"])
        

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        # encoder
        self.enc1 = conv_block(in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.enc2 = conv_block(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # decoder
        self.up = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)  # image size = (128*128) >>> (256, 256)

        self.dec = conv_block(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)  # combine feature channels = enc1(64)+up(64)

        self.final = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        pool = self.pool(enc1)
        enc2 = self.enc2(pool)

        up = self.up(enc2)

        tiger = torch.cat([up, enc1], dim=1)  # (Batch, Channel, Height, Width) so dim=1 代表合併的是C
        dec = self.dec(tiger)

        forward = self.final(dec)
        
        return forward