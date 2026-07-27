import numpy as np
from torch.utils.data import Sampler
from typing import List, Iterator

class FewShotSampler(Sampler):
    """N-way K-shot episode采样器

    每个episode随机选择N个类别，每个类别采样K个样本。

    Args:
        labels: 数据集中所有样本的标签列表
        nWay: 每个episode包含的类别数
        kShot: 每个类别的样本数（support set）
        nEpisodes: 总episode数量
        seed: 随机种子
    """

    def __init__(
        self,
        labels: List[int],
        nWay: int = 5,
        kShot: int = 5,
        nEpisodes: int = 100,
        seed: int = 42
    ):
        self.labels = np.array(labels)
        self.nWay = nWay
        self.kShot = kShot
        self.nEpisodes = nEpisodes
        self.rng = np.random.RandomState(seed)

        # 按类别组织样本索引
        self.class_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        # 检查每个类别是否有足够样本
        self.available_classes = []
        for cls, indices in self.class_to_indices.items():
            if len(indices) >= kShot:
                self.available_classes.append(cls)

        if len(self.available_classes) < nWay:
            raise ValueError(
                f"Not enough classes with >= {kShot} samples. "
                f"Found {len(self.available_classes)}, need {nWay}"
            )

    def __len__(self) -> int:
        return self.nEpisodes

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.nEpisodes):
            # 随机选择N个类别
            selected_classes = self.rng.choice(
                self.available_classes,
                size=self.nWay,
                replace=False
            )

            # 每个类别采样K个样本
            episode_indices = []
            for cls in selected_classes:
                cls_indices = self.class_to_indices[cls]
                selected = self.rng.choice(cls_indices, size=self.kShot, replace=False)
                episode_indices.extend(selected)

            yield episode_indices
