"""LoveDA数据集预处理transforms"""
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from typing import Tuple


class ResizePadding:
    """Resize保持宽高比，然后padding到目标尺寸

    Args:
        target_size: 目标尺寸（正方形）
        fill_value: 图像padding填充值（默认0，黑色）
        mask_fill_value: mask padding填充值（默认255，表示ignore）
    """

    def __init__(
        self,
        target_size: int = 1024,
        fill_value: int = 0,
        mask_fill_value: int = 255
    ):
        self.target_size = target_size
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """处理图像和mask

        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L mode)

        Returns:
            resized_image: (target_size, target_size, 3)
            resized_mask: (target_size, target_size)
        """
        w, h = image.size

        # 计算缩放比例（保持宽高比）
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize（图像用双线性，mask用最近邻）
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # 计算padding
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        # Padding
        image = ImageOps.expand(
            image,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill_value
        )
        mask = ImageOps.expand(
            mask,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=self.mask_fill_value
        )

        return image, mask


class DefaultTransform:
    """LoveDA数据集默认预处理pipeline"""

    def __init__(self, target_size: int = 1024):
        self.resize_padding = ResizePadding(target_size)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """完整预处理流程

        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L mode)

        Returns:
            image: (3, target_size, target_size) float32 tensor, normalized
            mask: (target_size, target_size) int64 tensor, values in [0, num_classes-1] or 255
        """
        # Resize + Padding
        image, mask = self.resize_padding(image, mask)

        # ToTensor
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        # Normalize (仅图像)
        image = self.normalize(image)

        return image, mask
