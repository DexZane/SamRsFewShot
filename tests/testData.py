import pytest
import numpy as np
from tests.fixtures.sampleData import create_sample_image, create_sample_dataset

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
    assert set(np.unique(mask)).issubset(set(range(7)))

def test_create_sample_dataset():
    """测试数据集生成"""
    dataset = create_sample_dataset(num_samples=10, numClasses=5)

    assert len(dataset) == 10
    for img, mask in dataset:
        assert img.shape == (512, 512, 3)
        assert mask.shape == (512, 512)
