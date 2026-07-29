"""
温和数据增强 - Phase 3.3
仅保留几何增强，移除颜色扰动
适合 few-shot 场景，保持 support-query 特征一致性
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random


class MildAugmentedTransform:
    """
    温和数据增强策略：
    1. 仅保留几何增强（翻转 + 小角度旋转）
    2. 移除颜色扰动（ColorJitter）
    3. 保持遥感影像的光谱特征完整性
    """

    def __init__(
        self,
        target_size: int = 1024,
        train: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotation_prob: float = 0.3,
        max_rotation: int = 15  # 小角度旋转
    ):
        self.target_size = target_size
        self.train = train

        if train:
            self.hflip_prob = hflip_prob
            self.vflip_prob = vflip_prob
            self.rotation_prob = rotation_prob
            self.max_rotation = max_rotation

    def __call__(self, image: Image.Image, mask: Image.Image):
        """
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L mode, 单通道)

        Returns:
            image_tensor: [3, 1024, 1024]
            mask_tensor: [1024, 1024]
        """
        # Step 1: Resize 保持宽高比 + Padding
        image, mask = self._resize_with_padding(image, mask)

        # Step 2: 训练时的温和增强
        if self.train:
            # 2.1 随机水平翻转
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 2.2 随机垂直翻转
            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 2.3 随机小角度旋转 (±15°)
            if random.random() < self.rotation_prob:
                angle = random.uniform(-self.max_rotation, self.max_rotation)
                image = TF.rotate(image, angle, fill=0)
                mask = TF.rotate(mask, angle, fill=0)

        # Step 3: ToTensor + Normalize
        image_tensor = TF.to_tensor(image)  # [3, H, W], [0, 1]
        mask_tensor = torch.from_numpy(np.array(mask)).long()  # [H, W]

        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor, mask_tensor

    def _resize_with_padding(self, image: Image.Image, mask: Image.Image):
        """
        保持宽高比的 resize + padding 到正方形
        """
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # Padding to square
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

        image = TF.pad(image, padding, fill=0)
        mask = TF.pad(mask, padding, fill=0)

        return image, mask
