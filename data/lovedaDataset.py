"""LoveDA遥感数据集加载器"""
import os
from typing import List, Tuple, Optional, Callable
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from .download import download_loveda, _verify_dataset
from .transforms import DefaultTransform


class LoveDADataset(Dataset):
    """LoveDA遥感语义分割数据集

    LoveDA包含城市和乡村场景的高分辨率遥感图像，共7个类别。

    Args:
        root: 数据集根目录
        split: 'Train', 'Val', 'Test'
        download: 是否自动下载数据集
        transform: 自定义transform（可选，默认使用DefaultTransform）

    Attributes:
        samples: List[(image_path, mask_path, class_id)]
        num_classes: 类别数（7）
        class_names: 类别名称列表
    """

    def __init__(
        self,
        root: str,
        split: str = 'Train',
        download: bool = True,
        transform: Optional[Callable] = None
    ):
        assert split in ['Train', 'Val', 'Test'], f"Invalid split: {split}"

        self.root = root
        self.split = split
        self.transform = transform if transform is not None else DefaultTransform()

        # 检查并下载数据集
        if download and not self._check_exists():
            download_loveda(root)

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found at {root}. "
                "Please set download=True or manually download the dataset."
            )

        # 扫描样本文件
        self.samples = self._load_samples()

        # 类别信息
        self.num_classes = 7
        self.class_names = [
            'background',
            'building',
            'road',
            'water',
            'barren',
            'forest',
            'agricultural'
        ]

    def _check_exists(self) -> bool:
        """检查数据集是否存在"""
        return _verify_dataset(self.root)

    def _load_samples(self) -> List[Tuple[str, str, int]]:
        """扫描并加载样本路径

        Returns:
            List of (image_path, mask_path, dominant_class_id)
        """
        samples = []

        # 遍历Urban和Rural
        for region in ['Urban', 'Rural']:
            images_dir = os.path.join(self.root, self.split, region, 'images_png')
            masks_dir = os.path.join(self.root, self.split, region, 'masks_png')

            if not os.path.isdir(images_dir):
                continue

            # 遍历图像文件
            image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

            for img_file in image_files:
                img_path = os.path.join(images_dir, img_file)
                mask_path = os.path.join(masks_dir, img_file)

                if not os.path.exists(mask_path):
                    continue

                # 读取mask获取主导类别（用于few-shot采样）
                mask = Image.open(mask_path)
                mask_array = np.array(mask, dtype=np.int64)
                mask_tensor = torch.from_numpy(mask_array)
                # 找到最常见的类别（排除255）
                unique, counts = torch.unique(mask_tensor[mask_tensor != 255], return_counts=True)
                if len(unique) > 0:
                    dominant_class = int(unique[counts.argmax()])
                else:
                    dominant_class = 0  # 默认为background

                samples.append((img_path, mask_path, dominant_class))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """加载单个样本

        Args:
            idx: 样本索引

        Returns:
            image: (3, 1024, 1024) float32 tensor, normalized
            mask: (1024, 1024) int64 tensor, values in [0, 6] or 255
            class_id: int, dominant class in mask
        """
        image_path, mask_path, class_id = self.samples[idx]

        # 读取图像和mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        # 应用transform
        image, mask = self.transform(image, mask)

        return image, mask, class_id
