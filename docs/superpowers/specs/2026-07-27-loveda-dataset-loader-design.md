# LoveDA数据集加载器设计文档

**日期**: 2026-07-27  
**项目**: SAM遥感少样本分割  
**阶段**: Phase 1  

## 1. 概述

为SAM遥感少样本分割项目实现LoveDA数据集加载器，支持自动下载、标准PyTorch Dataset接口、图像预处理和few-shot episode采样。

## 2. 设计决策

### 2.1 数据集获取方式
- **选择**: 自动下载（A方案）
- **理由**: 简化用户使用流程，提供完整的端到端体验
- **实现**: 使用conda环境，通过urllib/requests下载，tqdm显示进度

### 2.2 数据划分方式
- **选择**: 标准划分（A方案）
- **理由**: Phase 1作为基线实现，使用官方Train/Val/Test划分，结果更容易对比
- **详细**:
  - Train split → 用于训练（构建episodes）
  - Val split → 用于验证
  - Test split → 用于最终测试

### 2.3 图像预处理策略
- **选择**: 在线处理 + Padding保持宽高比（A方案 + 选项2）
- **理由**: 
  - 在线处理灵活，不占额外存储
  - Padding保持宽高比避免图像变形
  - 适合Mac本地开发测试
- **目标尺寸**: 1024×1024（SAM标准输入）

### 2.4 Episode数据组织
- **选择**: 单样本返回（A方案）
- **理由**: 
  - 职责清晰：Dataset负责加载单样本，Sampler负责组织episode
  - 符合PyTorch标准设计模式
  - 与现有FewShotSampler解耦

## 3. 架构设计

### 3.1 组件划分

```
LoveDADataset (data/lovedaDataset.py)
    ├── 负责单样本加载
    └── 提供标准PyTorch接口

download_loveda() (data/download.py)
    ├── 自动下载数据集
    └── 验证文件完整性

ResizePadding (data/transforms.py)
    ├── 保持宽高比resize
    └── Padding到1024×1024

FewShotSampler (已存在)
    └── Episode采样逻辑
```

### 3.2 数据流

```
用户代码
  ↓
LoveDADataset.__init__
  ├── 检查数据集是否存在
  ├── 不存在 → download_loveda()
  └── 扫描图像和mask文件
  ↓
LoveDADataset.__getitem__(idx)
  ├── 读取图像和mask (PIL Image)
  ├── ResizePadding(1024)
  ├── ToTensor + Normalize
  └── 返回 (image_tensor, mask_tensor, class_id)
  ↓
FewShotSampler
  ├── 根据类别标签组织episode
  └── 返回N×K个样本索引
  ↓
DataLoader
  └── Batch (N×K samples)
```

## 4. 核心类设计

### 4.1 LoveDADataset

```python
class LoveDADataset(Dataset):
    """LoveDA遥感数据集
    
    Args:
        root: 数据集根目录
        split: 'train', 'val', 'test'
        download: 是否自动下载
        transform: 自定义transform（可选）
    
    Attributes:
        samples: List[(image_path, mask_path, class_id)]
        num_classes: 类别数（7）
        class_names: 类别名称列表
    """
    
    def __init__(
        self, 
        root: str,
        split: str = 'train',
        download: bool = True,
        transform: Optional[Callable] = None
    ):
        self.root = root
        self.split = split
        self.transform = transform
        
        # 检查并下载数据集
        if download and not self._check_exists():
            download_loveda(root)
        
        # 扫描样本文件
        self.samples = self._load_samples()
        self.num_classes = 7
        self.class_names = [
            'background', 'building', 'road', 'water',
            'barren', 'forest', 'agricultural'
        ]
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        """加载单个样本
        
        Returns:
            image: (3, 1024, 1024) float32 tensor
            mask: (1024, 1024) long tensor
            class_id: int (dominant class in mask)
        """
        image_path, mask_path, class_id = self.samples[idx]
        
        # 读取图像和mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # 预处理
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # 默认预处理
            image, mask = self._default_transform(image, mask)
        
        return image, mask, class_id
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _check_exists(self) -> bool:
        """检查数据集是否存在"""
        pass
    
    def _load_samples(self) -> List[Tuple[str, str, int]]:
        """扫描并加载样本路径"""
        pass
    
    def _default_transform(self, image, mask):
        """默认预处理pipeline"""
        pass
```

### 4.2 下载功能

```python
def download_loveda(root: str) -> None:
    """自动下载LoveDA数据集
    
    Args:
        root: 下载目标目录
    
    Raises:
        RuntimeError: 下载或解压失败
    """
    import urllib.request
    import zipfile
    from tqdm import tqdm
    
    url = "https://zenodo.org/record/5706578/files/LoveDA.zip"
    zip_path = os.path.join(root, "LoveDA.zip")
    
    # 创建目录
    os.makedirs(root, exist_ok=True)
    
    # 下载（带进度条）
    print(f"Downloading LoveDA dataset from {url}...")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    except Exception as e:
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
        raise RuntimeError("Dataset structure verification failed")
    
    print("Download complete!")


def _verify_dataset(root: str) -> bool:
    """验证数据集文件结构完整性"""
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

### 4.3 图像预处理

```python
class ResizePadding:
    """Resize保持宽高比，然后padding到目标尺寸
    
    Args:
        target_size: 目标尺寸（正方形）
        fill_value: 图像padding填充值
        mask_fill_value: mask padding填充值（255表示ignore）
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
        
        Returns:
            resized_image: (target_size, target_size, 3)
            resized_mask: (target_size, target_size)
        """
        w, h = image.size
        
        # 计算缩放比例（保持宽高比）
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
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
        from PIL import ImageOps
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
    ) -> Tuple[Tensor, Tensor]:
        """完整预处理流程
        
        Returns:
            image: (3, 1024, 1024) float32 tensor, normalized
            mask: (1024, 1024) long tensor, values in [0, 6] or 255
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

## 5. 与FewShotSampler集成

### 5.1 使用示例

```python
# 创建数据集
dataset = LoveDADataset(
    root='./data/loveda',
    split='train',
    download=True
)

# 提取类别标签
labels = [dataset.samples[i][2] for i in range(len(dataset))]

# 创建FewShotSampler
sampler = FewShotSampler(
    labels=labels,
    nWay=5,
    kShot=5,
    nEpisodes=100
)

# 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=4
)

# 迭代episodes
for batch in dataloader:
    images, masks, labels = batch
    # images: (N*K, 3, 1024, 1024)
    # masks: (N*K, 1024, 1024)
    # labels: (N*K,)
```

### 5.2 Trainer中的episode重组

```python
# 在Trainer.train_epoch()中
for batch in dataloader:
    images, masks, labels = batch
    batch_size = images.shape[0]  # N*K
    
    # 重组为episode结构
    images = images.view(nWay, kShot, 3, 1024, 1024)
    masks = masks.view(nWay, kShot, 1024, 1024)
    labels = labels.view(nWay, kShot)
    
    # 分割support和query (如果需要)
    # support_images = images[:, :k_support, ...]
    # query_images = images[:, k_support:, ...]
```

## 6. 测试策略

### 6.1 单元测试

```python
# tests/test_loveda_dataset.py

def test_dataset_download():
    """测试自动下载功能"""
    with tempfile.TemporaryDirectory() as tmpdir:
        download_loveda(tmpdir)
        assert _verify_dataset(tmpdir)

def test_dataset_load():
    """测试数据加载"""
    dataset = LoveDADataset(root='./data/loveda', split='train')
    assert len(dataset) > 0
    
    image, mask, label = dataset[0]
    assert image.shape == (3, 1024, 1024)
    assert mask.shape == (1024, 1024)
    assert 0 <= label < 7

def test_padding_correctness():
    """测试padding正确性"""
    transform = ResizePadding(target_size=1024)
    
    # 测试宽图
    image = Image.new('RGB', (2048, 1024))
    mask = Image.new('L', (2048, 1024))
    img_out, mask_out = transform(image, mask)
    assert img_out.size == (1024, 1024)
    assert mask_out.size == (1024, 1024)
    
    # 测试高图
    image = Image.new('RGB', (512, 2048))
    mask = Image.new('L', (512, 2048))
    img_out, mask_out = transform(image, mask)
    assert img_out.size == (1024, 1024)
    assert mask_out.size == (1024, 1024)

def test_fewshot_integration():
    """测试与FewShotSampler集成"""
    dataset = LoveDADataset(root='./data/loveda', split='train')
    labels = [dataset.samples[i][2] for i in range(len(dataset))]
    
    sampler = FewShotSampler(labels, nWay=3, kShot=2, nEpisodes=5)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    
    batch = next(iter(dataloader))
    images, masks, labels = batch
    assert images.shape[0] == 6  # 3-way * 2-shot
    assert images.shape == (6, 3, 1024, 1024)
    assert masks.shape == (6, 1024, 1024)
```

### 6.2 可视化验证

```python
# scripts/visualize_dataset.py

import matplotlib.pyplot as plt

def visualize_samples(dataset, num_samples=5):
    """可视化数据集样本"""
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples*3))
    
    for i in range(num_samples):
        image, mask, label = dataset[i]
        
        # 反归一化图像
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Image (class: {label})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='tab10', vmin=0, vmax=6)
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('loveda_samples.png')
    print("Saved visualization to loveda_samples.png")
```

## 7. 文件结构

```
samRsFewShot/
├── data/
│   ├── __init__.py
│   ├── lovedaDataset.py      # 新增：LoveDADataset类
│   ├── download.py            # 新增：下载功能
│   ├── transforms.py          # 新增：预处理类
│   └── fewshotSampler.py      # 已存在
├── tests/
│   └── test_loveda_dataset.py # 新增：测试文件
├── scripts/
│   └── visualize_dataset.py   # 新增：可视化脚本
└── requirements.txt           # 更新：添加tqdm
```

## 8. 依赖更新

在 `requirements.txt` 中添加：
```
tqdm>=4.65.0
```

## 9. 实现优先级

1. **P0 - 核心功能**（必须）
   - LoveDADataset基础类
   - download_loveda()
   - ResizePadding
   - 基础测试

2. **P1 - 集成**（必须）
   - 与FewShotSampler集成测试
   - 完整的单元测试覆盖

3. **P2 - 增强**（可选）
   - 可视化脚本
   - 更详细的错误提示
   - 断点续传（下载）

## 10. 已知限制

1. **下载依赖网络**：国内可能需要配置代理或手动下载
2. **内存占用**：1024×1024图像较大，batch_size需要根据Mac内存调整
3. **类别不平衡**：LoveDA数据集可能存在类别不平衡，影响few-shot采样
4. **MPS兼容性**：部分PyTorch操作在MPS后端可能不支持，需要回退到CPU

## 11. 后续改进方向

- **Phase 2**: 支持数据增强（随机翻转、旋转、颜色抖动）
- **Phase 3**: 实现base/novel classes划分
- **Phase 4**: 添加缓存机制提升训练速度

---

**审核状态**: 待审核  
**下一步**: 调用 writing-plans 创建实现计划
