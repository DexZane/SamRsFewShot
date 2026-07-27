# Phase 2: 基线训练实验设置

## 当前状态

✅ **Phase 1 完成**：
- LoveDA数据集加载器实现完成
- 12个单元测试全部通过
- FewShotSampler集成验证通过
- 数据预处理pipeline完成

⏸️ **Phase 2 待启动**：基线训练实验

## 前置准备

### 1. 下载LoveDA数据集

LoveDA数据集需要手动下载：

**数据集信息**：
- 来源：Zenodo
- 链接：https://zenodo.org/record/5706578
- DOI：10.5281/zenodo.5706578
- 大小：约 2.5 GB

**下载步骤**：
1. 访问 https://zenodo.org/record/5706578
2. 下载 `LoveDA.zip` 文件
3. 解压到 `./data/LoveDA/` 目录

**预期目录结构**：
```
data/LoveDA/
├── Train/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
├── Val/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
└── Test/
    ├── Urban/
    │   ├── images_png/
    │   └── masks_png/
    └── Rural/
        ├── images_png/
        └── masks_png/
```

### 2. 下载SAM预训练模型

**SAM模型下载**：
- 模型：ViT-B (Base)
- 链接：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- 大小：约 375 MB

**下载命令**：
```bash
mkdir -p ./checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O ./checkpoints/sam_vit_b_01ec64.pth
```

或使用curl：
```bash
mkdir -p ./checkpoints
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o ./checkpoints/sam_vit_b_01ec64.pth
```

## Phase 2 训练实验

### 实验目标

1. **基线实验**：验证SAM + LoRA + Prompt Learning在LoveDA数据集上的性能
2. **Few-shot设置**：5-way 5-shot语义分割
3. **评估指标**：mIoU (mean Intersection over Union)

### 训练命令

```bash
python scripts/train.py \
    --dataRoot ./data/LoveDA \
    --samCheckpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --samModelType vit_b \
    --loraRank 4 \
    --nPrompts 5 \
    --nWay 5 \
    --kShot 5 \
    --numEpochs 50 \
    --batchSize 4 \
    --lr 1e-4 \
    --device mps \
    --evalInterval 5 \
    --saveInterval 10 \
    --checkpointDir ./checkpoints \
    --logDir ./runs
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataRoot` | `./data/LoveDA` | 数据集根目录 |
| `--samCheckpoint` | 必需 | SAM预训练模型路径 |
| `--samModelType` | `vit_b` | SAM模型类型 (vit_b/vit_l/vit_h) |
| `--loraRank` | 4 | LoRA秩 |
| `--nPrompts` | 5 | 每类可学习提示数量 |
| `--nWay` | 5 | Few-shot类别数 |
| `--kShot` | 5 | 每类支持样本数 |
| `--numEpochs` | 50 | 训练轮数 |
| `--batchSize` | 4 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--device` | `mps` | 训练设备 (mps/cuda/cpu) |
| `--evalInterval` | 5 | 评估间隔 |
| `--saveInterval` | 10 | 保存间隔 |

### 预期输出

训练过程会生成：

1. **Checkpoints**：保存在 `./checkpoints/` 目录
   - `checkpoint_epoch_10.pth`
   - `checkpoint_epoch_20.pth`
   - `best_model.pth` (最佳mIoU模型)

2. **TensorBoard日志**：保存在 `./runs/` 目录
   - 训练损失曲线
   - 验证mIoU曲线
   - 学习率变化

3. **控制台输出**：
   - 每个epoch的训练损失
   - 每5个epoch的验证mIoU
   - 模型参数统计

### 可视化训练过程

启动TensorBoard：
```bash
tensorboard --logdir=./runs
```

访问：http://localhost:6006

## Phase 2 验证标准

### 阶段门槛（Phase 1 → Phase 2）

✅ **已满足**：
- [x] 所有代码模块实现
- [x] 单元测试100%通过
- [x] 核心功能手动验证
- [x] 文档完整

### Phase 2 完成标准

需要满足以下条件才算完成Phase 2：

- [ ] 基线实验完成（至少1次完整训练）
- [ ] 有mIoU指标数据
- [ ] 训练曲线收敛
- [ ] 实验记录完整（保存在项目文档中）

### 实验记录模板

在 `docs/experiments/` 目录创建实验记录：

```markdown
# Experiment 1: Baseline Training

## 配置
- 数据集：LoveDA (Train/Val)
- 模型：SAM-ViT-B + LoRA (rank=4) + Prompt Learning
- Few-shot设置：5-way 5-shot
- 训练轮数：50 epochs
- 学习率：1e-4

## 结果
- 最佳验证mIoU：XX.XX%
- 训练时间：XX小时
- GPU显存占用：XX GB

## 分析
- 训练曲线是否收敛？
- 与预期性能对比？
- 存在的问题？

## 下一步
- 改进方向
- 超参数调整
```

## 注意事项

1. **设备要求**：
   - Mac (Apple Silicon) 使用 `--device mps`
   - NVIDIA GPU 使用 `--device cuda`
   - CPU训练较慢，建议仅用于调试

2. **内存需求**：
   - 训练需要至少16GB RAM
   - GPU/MPS显存需要至少8GB

3. **训练时间估算**：
   - Mac (M1/M2) MPS：约4-6小时 (50 epochs)
   - NVIDIA GPU (V100)：约2-3小时
   - CPU：约20-30小时（不推荐）

4. **故障排查**：
   - 如果遇到OOM（内存不足），减小 `--batchSize`
   - 如果训练不收敛，调整 `--lr`
   - 如果数据加载慢，检查数据集路径

## 快速启动脚本

创建了 `scripts/run_phase2_baseline.sh` 快速启动脚本：

```bash
bash scripts/run_phase2_baseline.sh
```

该脚本会自动检查前置条件并启动训练。
