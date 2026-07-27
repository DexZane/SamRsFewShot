import pytest
import numpy as np
from collections import Counter
from tests.fixtures.sampleData import create_sample_image, create_sample_dataset
from data.fewshotSampler import FewShotSampler

def test_create_sample_image():
    """测试单个样本生成"""
    img, mask = create_sample_image(size=(512, 512), numClasses=7)

    # 检查图像
    assert img.shape == (512, 512, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255

    # 检查掩码
    assert mask.shape == (512, 512)
    assert mask.dtype == np.int64
    assert set(np.unique(mask)) == set(range(7))

def test_create_sample_dataset():
    """测试数据集生成"""
    dataset = create_sample_dataset(num_samples=10, numClasses=5)

    assert len(dataset) == 10
    for img, mask in dataset:
        assert img.shape == (512, 512, 3)
        assert mask.shape == (512, 512)

def test_fewshot_sampler():
    """测试N-way K-shot采样器"""
    # 模拟标签数据：10个类别，每类10个样本
    labels = []
    for cls in range(10):
        labels.extend([cls] * 10)

    sampler = FewShotSampler(
        labels=labels,
        nWay=5,
        kShot=3,
        nEpisodes=10
    )

    # 测试采样器长度
    assert len(sampler) == 10

    # 测试每个episode
    for episode_indices in sampler:
        # 应该采样 5 classes × 3 shots = 15 个样本
        assert len(episode_indices) == 15

        # 检查是否来自5个不同类别
        episode_labels = [labels[i] for i in episode_indices]
        unique_labels = set(episode_labels)
        assert len(unique_labels) == 5

        # 检查每个类别恰好3个样本
        label_counts = Counter(episode_labels)
        assert all(count == 3 for count in label_counts.values())
