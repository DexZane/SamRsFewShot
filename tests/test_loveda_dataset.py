import os
import pytest
import torch
from PIL import Image
import tempfile
from data.lovedaDataset import LoveDADataset
from torch.utils.data import DataLoader
from data.fewshotSampler import FewShotSampler


@pytest.fixture
def mock_dataset_structure():
    """创建mock数据集结构用于测试"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建目录结构
        for split in ['Train', 'Val', 'Test']:
            for region in ['Urban', 'Rural']:
                images_dir = os.path.join(tmpdir, split, region, 'images_png')
                masks_dir = os.path.join(tmpdir, split, region, 'masks_png')
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(masks_dir, exist_ok=True)

                # 创建2个样本文件
                for i in range(2):
                    # 创建图像
                    img = Image.new('RGB', (512, 512), color=(i*50, i*50, i*50))
                    img.save(os.path.join(images_dir, f'{i}.png'))

                    # 创建mask（随机类别0-6）
                    mask = Image.new('L', (512, 512), color=i % 7)
                    mask.save(os.path.join(masks_dir, f'{i}.png'))

        yield tmpdir


def test_dataset_init_no_download(mock_dataset_structure):
    """测试Dataset初始化（数据集已存在）"""
    dataset = LoveDADataset(
        root=mock_dataset_structure,
        split='Train',
        download=False
    )

    assert dataset.num_classes == 7
    assert len(dataset.class_names) == 7
    assert len(dataset) > 0


def test_dataset_getitem_shapes(mock_dataset_structure):
    """测试__getitem__返回的数据形状"""
    dataset = LoveDADataset(
        root=mock_dataset_structure,
        split='Train',
        download=False
    )

    image, mask, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(label, int)

    assert image.shape == (3, 1024, 1024)
    assert mask.shape == (1024, 1024)
    assert 0 <= label < 7


def test_dataset_different_splits(mock_dataset_structure):
    """测试不同split的数据集"""
    train_dataset = LoveDADataset(root=mock_dataset_structure, split='Train', download=False)
    val_dataset = LoveDADataset(root=mock_dataset_structure, split='Val', download=False)
    test_dataset = LoveDADataset(root=mock_dataset_structure, split='Test', download=False)

    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(test_dataset) > 0


def test_fewshot_integration(mock_dataset_structure):
    """测试与FewShotSampler集成"""
    dataset = LoveDADataset(
        root=mock_dataset_structure,
        split='Train',
        download=False
    )

    # 提取类别标签
    labels = [dataset.samples[i][2] for i in range(len(dataset))]

    # 创建FewShotSampler（使用较小的参数）
    sampler = FewShotSampler(
        labels=labels,
        nWay=2,  # 2-way
        kShot=2,  # 2-shot
        nEpisodes=3
    )

    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0
    )

    # 测试迭代
    batch = next(iter(dataloader))
    images, masks, labels_batch = batch

    assert images.shape[0] == 4  # 2-way * 2-shot = 4
    assert images.shape == (4, 3, 1024, 1024)
    assert masks.shape == (4, 1024, 1024)
    assert labels_batch.shape == (4,)


def test_dataloader_multiple_episodes(mock_dataset_structure):
    """测试多个episode迭代"""
    dataset = LoveDADataset(
        root=mock_dataset_structure,
        split='Train',
        download=False
    )

    labels = [dataset.samples[i][2] for i in range(len(dataset))]

    sampler = FewShotSampler(
        labels=labels,
        nWay=2,
        kShot=1,
        nEpisodes=5
    )

    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    episode_count = 0
    for batch in dataloader:
        images, masks, labels_batch = batch
        assert images.shape[0] == 2  # 2-way * 1-shot
        episode_count += 1

    assert episode_count == 5
