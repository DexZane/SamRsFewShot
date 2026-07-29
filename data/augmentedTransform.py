"""数据增强版 Transform for Phase 3.1"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Tuple
import random

from data.transforms import ResizePadding


class AugmentedTransform:
    """增强版数据预处理，添加随机翻转、旋转、颜色抖动"""

    def __init__(
        self,
        target_size: int = 1024,
        train: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        color_jitter_prob: float = 0.5
    ):
        """
        Args:
            target_size: 目标尺寸
            train: 训练模式（True）或验证模式（False）
            hflip_prob: 水平翻转概率
            vflip_prob: 垂直翻转概率
            rotation_prob: 旋转概率（90/180/270度）
            color_jitter_prob: 颜色抖动概率
        """
        self.target_size = target_size
        self.train = train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_prob = rotation_prob
        self.color_jitter_prob = color_jitter_prob

        self.resize_padding = ResizePadding(target_size)

        # 颜色抖动（仅训练时）
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L mode)

        Returns:
            image: (3, H, W) normalized tensor
            mask: (H, W) int64 tensor
        """
        # 1. Resize + Padding
        image, mask = self.resize_padding(image, mask)

        # 2. 数据增强（仅训练模式）
        if self.train:
            # 随机水平翻转
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 随机垂直翻转
            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 随机旋转90/180/270度
            if random.random() < self.rotation_prob:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            # 颜色抖动（仅图像）
            if random.random() < self.color_jitter_prob:
                image = self.color_jitter(image)

        # 3. ToTensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        # 4. Normalize
        image = self.normalize(image)

        return image, mask
