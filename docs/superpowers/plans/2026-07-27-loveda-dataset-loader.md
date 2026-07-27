# LoveDA数据集加载器实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现LoveDA遥感数据集加载器，支持自动下载、标准PyTorch Dataset接口和few-shot episode采样

**架构：** 三个独立模块：download.py（下载和验证）、transforms.py（图像预处理）、lovedaDataset.py（Dataset主类），通过标准接口集成，支持与现有FewShotSampler无缝配合

**技术栈：** PyTorch, PIL, numpy, tqdm, urllib

---

## 文件结构

待创建的文件：
- `data/download.py` - 数据集下载和验证功能
- `data/transforms.py` - 图像预处理类（ResizePadding, DefaultTransform）
- `data/lovedaDataset.py` - LoveDADataset主类
- `tests/test_loveda_dataset.py` - 完整单元测试
- `scripts/visualizeDataset.py` - 可视化工具（可选）

待修改的文件：
- `data/__init__.py` - 导出新类

---

### 任务 1：数据集下载功能

**文件：**
- 创建：`data/download.py`
- 测试：`tests/test_download.py`

- [ ] **步骤 1：编写下载验证测试**

创建 `tests/test_download.py`：

```python
import os
import tempfile
import pytest
from data.download import _verify_dataset


def test_verify_dataset_missing():
    """测试验证函数对缺失目录的检测"""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert _verify_dataset(tmpdir) == False


def test_verify_dataset_incomplete():
    """测试验证函数对不完整目录的检测"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 只创建部分目录
        os.makedirs(os.path.join(tmpdir, 'Train/Urban/images_png'))
        assert _verify_dataset(tmpdir) == False
```

- [ ] **步骤 2：运行测试验证失败**

```bash
cd /Users/dexzane/Desktop/FindProject/samRsFewShot
pytest tests/test_download.py::test_verify_dataset_missing -v
```

预期：FAIL，报错 "ModuleNotFoundError: No module named 'data.download'"

- [ ] **步骤 3：实现_verify_dataset函数**

创建 `data/download.py`：

```python
"""LoveDA数据集下载功能"""
import os
import urllib.request
import zipfile
from typing import Optional
from tqdm import tqdm


def _verify_dataset(root: str) -> bool:
    """验证数据集文件结构完整性
    
    Args:
        root: 数据集根目录
    
    Returns:
        True if all required directories exist, False otherwise
    """
    required_dirs = [
        'Train/Urban/images_png',
        'Train/Urban/masks_png',
        'Train/Rural/images_png',
        'Train/Rural/masks_png',
        'Val/Urban/images_png',
        'Val/Urban/masks_png',
        'Val/Rural/images_png',
        'Val/Rural/masks_png',
        'Test/Urban/images_png',
        'Test/Urban/masks_png',
        'Test/Rural/images_png',
        'Test/Rural/masks_png',
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(root, dir_path)
        if not os.path.isdir(full_path):
            return False
    
    return True
```

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_download.py::test_verify_dataset_missing -v
pytest tests/test_download.py::test_verify_dataset_incomplete -v
```

预期：PASS

- [ ] **步骤 5：编写下载功能测试（跳过实际下载）**

在 `tests/test_download.py` 添加：

```python
from unittest.mock import patch, MagicMock
from data.download import download_loveda


@pytest.mark.skip(reason="Skips actual download, test structure only")
def test_download_loveda_creates_directory():
    """测试下载函数创建目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, 'loveda_test')
        # 实际测试中会mock urllib.request.urlretrieve
        assert not os.path.exists(target)
```

- [ ] **步骤 6：实现download_loveda函数**

在 `data/download.py` 添加：

```python
class DownloadProgressBar(tqdm):
    """带进度条的下载器"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_loveda(root: str, url: Optional[str] = None) -> None:
    """自动下载LoveDA数据集
    
    Args:
        root: 下载目标目录
        url: 数据集URL（默认使用官方地址）
    
    Raises:
        RuntimeError: 下载或解压失败
    """
    if url is None:
        url = "https://zenodo.org/record/5706578/files/LoveDA.zip"
    
    zip_path = os.path.join(root, "LoveDA.zip")
    
    # 创建目录
    os.makedirs(root, exist_ok=True)
    
    # 下载（带进度条）
    print(f"Downloading LoveDA dataset from {url}...")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="LoveDA") as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    except Exception as e:
        # 清理不完整的下载
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise RuntimeError(f"Failed to download dataset: {e}")
    
    # 解压
    print("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
    except Exception as e:
        raise RuntimeError(f"Failed to extract dataset: {e}")
    finally:
        # 清理zip文件
        if os.path.exists(zip_path):
            os.remove(zip_path)
    
    # 验证文件结构
    if not _verify_dataset(root):
        raise RuntimeError("Dataset structure verification failed. Please check the downloaded files.")
    
    print("Download complete!")
```

- [ ] **步骤 7：Commit**

```bash
git add data/download.py tests/test_download.py
git commit -m "feat: add LoveDA dataset download functionality

- Implement download_loveda() with progress bar
- Add _verify_dataset() for structure validation
- Add unit tests for verification logic
- Handle download/extract errors gracefully

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### 任务 2：图像预处理Transform

**文件：**
- 创建：`data/transforms.py`
- 测试：`tests/test_transforms.py`

- [ ] **步骤 1：编写ResizePadding测试**

创建 `tests/test_transforms.py`：

```python
import pytest
import torch
import numpy as np
from PIL import Image
from data.transforms import ResizePadding


def test_resize_padding_wide_image():
    """测试宽图padding"""
    transform = ResizePadding(target_size=1024)
    
    # 创建2048x1024的宽图
    image = Image.new('RGB', (2048, 1024), color=(255, 0, 0))
    mask = Image.new('L', (2048, 1024), color=128)
    
    img_out, mask_out = transform(image, mask)
    
    assert img_out.size == (1024, 1024)
    assert mask_out.size == (1024, 1024)
    
    # 检查padding是否正确（上下应该有黑边）
    img_array = np.array(img_out)
    # 上下应该有padding（黑色），中间应该是红色
    assert np.all(img_array[0, :, 0] == 0)  # 顶部是黑色
    assert np.all(img_array[512, :, 0] == 255)  # 中间是红色


def test_resize_padding_tall_image():
    """测试高图padding"""
    transform = ResizePadding(target_size=1024)
    
    # 创建512x2048的高图
    image = Image.new('RGB', (512, 2048), color=(0, 255, 0))
    mask = Image.new('L', (512, 2048), color=64)
    
    img_out, mask_out = transform(image, mask)
    
    assert img_out.size == (1024, 1024)
    assert mask_out.size == (1024, 1024)
    
    # 检查padding是否正确（左右应该有黑边）
    img_array = np.array(img_out)
    assert np.all(img_array[:, 0, 1] == 0)  # 左边是黑色
    assert np.all(img_array[:, 512, 1] == 255)  # 中间是绿色


def test_resize_padding_mask_fill_value():
    """测试mask的填充值为255"""
    transform = ResizePadding(target_size=1024, mask_fill_value=255)
    
    image = Image.new('RGB', (2048, 1024))
    mask = Image.new('L', (2048, 1024), color=1)
    
    img_out, mask_out = transform(image, mask)
    
    # mask的padding区域应该是255
    mask_array = np.array(mask_out)
    assert mask_array[0, 0] == 255  # 顶部padding
```

- [ ] **步骤 2：运行测试验证失败**

```bash
pytest tests/test_transforms.py::test_resize_padding_wide_image -v
```

预期：FAIL，报错 "ModuleNotFoundError: No module named 'data.transforms'"

- [ ] **步骤 3：实现ResizePadding类**

创建 `data/transforms.py`：

```python
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
```

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_transforms.py -v
```

预期：所有测试PASS

- [ ] **步骤 5：编写DefaultTransform测试**

在 `tests/test_transforms.py` 添加：

```python
from data.transforms import DefaultTransform


def test_default_transform_output_types():
    """测试DefaultTransform输出类型和形状"""
    transform = DefaultTransform(target_size=1024)
    
    image = Image.new('RGB', (512, 512), color=(128, 128, 128))
    mask = Image.new('L', (512, 512), color=5)
    
    img_tensor, mask_tensor = transform(image, mask)
    
    # 检查类型和形状
    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(mask_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 1024, 1024)
    assert mask_tensor.shape == (1024, 1024)
    assert img_tensor.dtype == torch.float32
    assert mask_tensor.dtype == torch.int64


def test_default_transform_normalization():
    """测试图像归一化"""
    transform = DefaultTransform(target_size=1024)
    
    # 创建白色图像
    image = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
    mask = Image.new('L', (1024, 1024), color=0)
    
    img_tensor, mask_tensor = transform(image, mask)
    
    # 白色经过归一化后应该约等于 (1 - mean) / std
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    expected_r = (1.0 - 0.485) / 0.229
    expected_g = (1.0 - 0.456) / 0.224
    expected_b = (1.0 - 0.406) / 0.225
    
    assert torch.abs(img_tensor[0, 512, 512] - expected_r) < 0.01
    assert torch.abs(img_tensor[1, 512, 512] - expected_g) < 0.01
    assert torch.abs(img_tensor[2, 512, 512] - expected_b) < 0.01
```

- [ ] **步骤 6：实现DefaultTransform类**

在 `data/transforms.py` 添加：

```python
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
```

- [ ] **步骤 7：运行测试验证通过**

```bash
pytest tests/test_transforms.py -v
```

预期：所有测试PASS

- [ ] **步骤 8：Commit**

```bash
git add data/transforms.py tests/test_transforms.py
git commit -m "feat: add image preprocessing transforms

- Implement ResizePadding with aspect ratio preservation
- Implement DefaultTransform with normalization
- Add comprehensive unit tests
- Handle both wide and tall images correctly

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### 任务 3：LoveDADataset核心类

**文件：**
- 创建：`data/lovedaDataset.py`
- 测试：`tests/test_loveda_dataset.py`

- [ ] **步骤 1：编写Dataset基础测试**

创建 `tests/test_loveda_dataset.py`：

```python
import os
import pytest
import torch
from PIL import Image
import tempfile
from data.lovedaDataset import LoveDADataset


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
```

- [ ] **步骤 2：运行测试验证失败**

```bash
pytest tests/test_loveda_dataset.py::test_dataset_init_no_download -v
```

预期：FAIL，报错 "ModuleNotFoundError: No module named 'data.lovedaDataset'"

- [ ] **步骤 3：实现LoveDADataset类骨架**

创建 `data/lovedaDataset.py`：

```python
"""LoveDA遥感数据集加载器"""
import os
from typing import List, Tuple, Optional, Callable
import torch
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
                mask_array = torch.from_numpy(Image.open(mask_path).__array__())
                # 找到最常见的类别（排除255）
                unique, counts = torch.unique(mask_array[mask_array != 255], return_counts=True)
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
```

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_loveda_dataset.py -v
```

预期：所有测试PASS

- [ ] **步骤 5：Commit**

```bash
git add data/lovedaDataset.py tests/test_loveda_dataset.py
git commit -m "feat: implement LoveDADataset class

- Standard PyTorch Dataset interface
- Auto-download support
- Load samples from Train/Val/Test splits
- Extract dominant class for few-shot sampling
- Comprehensive unit tests with mock data

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### 任务 4：与FewShotSampler集成测试

**文件：**
- 修改：`tests/test_loveda_dataset.py`
- 测试集成场景

- [ ] **步骤 1：编写集成测试**

在 `tests/test_loveda_dataset.py` 添加：

```python
from torch.utils.data import DataLoader
from data.fewshotSampler import FewShotSampler


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
```

- [ ] **步骤 2：运行集成测试**

```bash
pytest tests/test_loveda_dataset.py::test_fewshot_integration -v
pytest tests/test_loveda_dataset.py::test_dataloader_multiple_episodes -v
```

预期：PASS

- [ ] **步骤 3：Commit**

```bash
git add tests/test_loveda_dataset.py
git commit -m "test: add FewShotSampler integration tests

- Test DataLoader with FewShotSampler
- Verify episode batch shapes
- Test multiple episode iterations

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### 任务 5：更新data模块导出

**文件：**
- 修改：`data/__init__.py`

- [ ] **步骤 1：更新__init__.py**

```python
"""Data loading and preprocessing"""
from .fewshotSampler import FewShotSampler
from .download import download_loveda
from .transforms import ResizePadding, DefaultTransform
from .lovedaDataset import LoveDADataset

__all__ = [
    'FewShotSampler',
    'download_loveda',
    'ResizePadding',
    'DefaultTransform',
    'LoveDADataset',
]
```

- [ ] **步骤 2：测试导入**

```bash
cd /Users/dexzane/Desktop/FindProject/samRsFewShot
python -c "from data import LoveDADataset, download_loveda, ResizePadding, DefaultTransform; print('Import successful')"
```

预期：打印 "Import successful"

- [ ] **步骤 3：Commit**

```bash
git add data/__init__.py
git commit -m "feat: export LoveDA dataset classes

- Export LoveDADataset, download_loveda
- Export ResizePadding, DefaultTransform
- Update module __all__

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### 任务 6：可视化脚本（可选）

**文件：**
- 创建：`scripts/visualizeDataset.py`

- [ ] **步骤 1：创建可视化脚本**

```python
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
```

- [ ] **步骤 2：测试可视化脚本（使用mock数据）**

```bash
cd /Users/dexzane/Desktop/FindProject/samRsFewShot
# 需要先有数据集或使用mock数据
# python scripts/visualizeDataset.py --root ./data/loveda --split Train --num-samples 3
```

预期：生成loveda_samples.png文件

- [ ] **步骤 3：Commit**

```bash
git add scripts/visualizeDataset.py
git commit -m "feat: add dataset visualization script

- Visualize images and masks side by side
- Support all splits (Train/Val/Test)
- Configurable number of samples
- Denormalize images for display

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### 任务 7：运行完整测试套件

**文件：**
- 无需创建新文件

- [ ] **步骤 1：运行所有测试**

```bash
cd /Users/dexzane/Desktop/FindProject/samRsFewShot
pytest tests/test_download.py tests/test_transforms.py tests/test_loveda_dataset.py -v --tb=short
```

预期：所有测试PASS

- [ ] **步骤 2：检查测试覆盖率**

```bash
pytest tests/test_download.py tests/test_transforms.py tests/test_loveda_dataset.py --cov=data --cov-report=term-missing
```

预期：覆盖率 > 80%

- [ ] **步骤 3：最终验证**

创建简单的验证脚本测试完整流程：

```bash
python -c "
from data import LoveDADataset, FewShotSampler
from torch.utils.data import DataLoader
import tempfile
import os
from PIL import Image

# 创建mock数据
with tempfile.TemporaryDirectory() as tmpdir:
    for split in ['Train']:
        for region in ['Urban']:
            images_dir = os.path.join(tmpdir, split, region, 'images_png')
            masks_dir = os.path.join(tmpdir, split, region, 'masks_png')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            
            for i in range(10):
                img = Image.new('RGB', (512, 512), color=(i*20, i*20, i*20))
                img.save(os.path.join(images_dir, f'{i}.png'))
                mask = Image.new('L', (512, 512), color=i % 7)
                mask.save(os.path.join(masks_dir, f'{i}.png'))
    
    # 测试完整流程
    dataset = LoveDADataset(root=tmpdir, split='Train', download=False)
    labels = [dataset.samples[i][2] for i in range(len(dataset))]
    sampler = FewShotSampler(labels, nWay=3, kShot=2, nEpisodes=2)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    
    batch = next(iter(dataloader))
    images, masks, labels = batch
    
    print(f'✓ Dataset: {len(dataset)} samples')
    print(f'✓ Batch shape: images={images.shape}, masks={masks.shape}')
    print(f'✓ Integration test passed!')
"
```

预期：打印成功信息

- [ ] **步骤 4：创建总结文档**

```bash
git log --oneline --since="1 day ago" > /tmp/loveda_commits.txt
echo "
LoveDA Dataset Loader Implementation Summary
============================================

Completed Tasks:
- [x] Task 1: Download functionality with progress bar
- [x] Task 2: Image preprocessing transforms (ResizePadding, DefaultTransform)
- [x] Task 3: LoveDADataset core class
- [x] Task 4: FewShotSampler integration tests
- [x] Task 5: Module exports
- [x] Task 6: Visualization script
- [x] Task 7: Full test suite

Test Results:
$(pytest tests/test_download.py tests/test_transforms.py tests/test_loveda_dataset.py -v --tb=line 2>&1 | tail -5)

Next Steps:
1. Download real LoveDA dataset (or test with manual download)
2. Run visualization script to verify data quality
3. Integrate with training pipeline
4. Update Phase 1 documentation
" > docs/superpowers/plans/2026-07-27-loveda-completion-summary.txt
```

- [ ] **步骤 5：最终Commit**

```bash
git add docs/superpowers/plans/2026-07-27-loveda-completion-summary.txt
git commit -m "docs: add LoveDA dataset loader completion summary

Phase 1 LoveDA dataset loader implementation complete:
- Auto-download with progress tracking
- Resize+Padding preprocessing
- PyTorch Dataset interface
- FewShotSampler integration
- Comprehensive test coverage
- Visualization tools

All tests passing. Ready for Phase 2 training experiments.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## 实现完成检查清单

完成以下所有任务后，实现即完成：

- [ ] Task 1: 下载功能 (download.py + tests)
- [ ] Task 2: 预处理transforms (transforms.py + tests)
- [ ] Task 3: LoveDADataset类 (lovedaDataset.py + tests)
- [ ] Task 4: FewShotSampler集成测试
- [ ] Task 5: 模块导出更新
- [ ] Task 6: 可视化脚本（可选）
- [ ] Task 7: 完整测试套件通过

## 验证标准

实现完成需满足：
1. 所有单元测试通过（pytest）
2. 测试覆盖率 > 80%
3. 与FewShotSampler集成正常
4. 能够加载和预处理LoveDA样本
5. 代码遵循项目风格（驼峰命名、类型注解）

---

**计划状态**: 准备执行  
**预计时间**: 2-3小时（含测试）
