"""可视化LoveDA数据集样本"""
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from data import LoveDADataset


def visualize_samples(dataset, num_samples=5, save_path='loveda_samples.png'):
    """可视化数据集样本

    Args:
        dataset: LoveDADataset实例
        num_samples: 可视化样本数量
        save_path: 保存路径
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples*3))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        if i >= len(dataset):
            break

        image, mask, label = dataset[i]

        # 反归一化图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)

        # 显示图像
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Image (class: {dataset.class_names[label]})')
        axes[i, 0].axis('off')

        # 显示mask
        mask_np = mask.numpy()
        axes[i, 1].imshow(mask_np, cmap='tab10', vmin=0, vmax=6)
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize LoveDA dataset samples')
    parser.add_argument('--root', type=str, default='./data/loveda',
                        help='Dataset root directory')
    parser.add_argument('--split', type=str, default='Train',
                        choices=['Train', 'Val', 'Test'],
                        help='Dataset split')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default='loveda_samples.png',
                        help='Output file path')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset if not exists')

    args = parser.parse_args()

    # 创建数据集
    print(f"Loading {args.split} dataset from {args.root}...")
    dataset = LoveDADataset(
        root=args.root,
        split=args.split,
        download=args.download
    )

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Classes: {dataset.class_names}")

    # 可视化
    visualize_samples(dataset, args.num_samples, args.output)


if __name__ == '__main__':
    main()
