import numpy as np
import torch
from pathlib import Path

def create_sample_image(size=(512, 512), numClasses=7):
    """生成模拟遥感图像

    Args:
        size: 图像尺寸 (H, W)
        numClasses: 类别数量

    Returns:
        image: RGB图像 (H, W, 3), uint8
        mask: 语义掩码 (H, W), int64
    """
    H, W = size

    # 生成RGB图像（模拟遥感影像）
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    # 生成语义掩码（每个类别占一块区域）
    mask = np.zeros((H, W), dtype=np.int64)
    patch_h = H // numClasses
    for i in range(numClasses):
        start = i * patch_h
        end = H if i == numClasses - 1 else (i + 1) * patch_h
        mask[start:end, :] = i

    return image, mask

def create_sample_dataset(num_samples=20, numClasses=7, save_dir=None):
    """生成测试数据集

    Args:
        num_samples: 样本数量
        numClasses: 类别数量
        save_dir: 保存路径（可选）

    Returns:
        dataset: list of (image, mask) tuples
    """
    dataset = []
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        img, mask = create_sample_image(numClasses=numClasses)
        dataset.append((img, mask))

        if save_dir:
            np.save(save_dir / f"img_{i:03d}.npy", img)
            np.save(save_dir / f"mask_{i:03d}.npy", mask)

    return dataset
