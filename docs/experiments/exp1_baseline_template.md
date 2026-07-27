# Experiment 1: Baseline Training

**实验日期**: 待填写  
**实验者**: 待填写  
**状态**: 待执行

## 实验目标

验证SAM + LoRA + Prompt Learning在LoveDA数据集上的基线性能，为后续改进提供对比基准。

## 配置

### 数据集
- **数据集名称**: LoveDA Remote Sensing Dataset
- **类别数**: 7类 (background, building, road, water, barren, forest, agricultural)
- **训练集**: Train split (Urban + Rural)
- **验证集**: Val split (Urban + Rural)
- **预处理**: 
  - Resize with aspect ratio preservation
  - Padding to 1024×1024
  - ImageNet normalization

### 模型配置
- **基础模型**: SAM-ViT-B (Segment Anything Model)
- **参数高效适配**: LoRA (rank=4, alpha=8, dropout=0.1)
- **Prompt Learning**: 每类5个可学习提示向量

### Few-shot设置
- **N-way**: 5 (每个episode选5个类别)
- **K-shot**: 5 (每个类别5个support样本)
- **Query samples**: 15个/类 (训练), 10个/类 (验证)
- **Episodes**: 100 (训练), 20 (验证)

### 训练参数
- **Epochs**: 50
- **Batch size**: 4
- **Learning rate**: 1e-4
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR
- **Device**: MPS (Apple Silicon) / CUDA (NVIDIA GPU)

## 结果

### 定量结果

| Metric | Value | Epoch |
|--------|-------|-------|
| Best Val mIoU | 待填写 | 待填写 |
| Final Train Loss | 待填写 | 50 |
| Final Val Loss | 待填写 | 50 |
| Training Time | 待填写 | - |
| GPU Memory Usage | 待填写 | - |

### 类别级别结果

| Class | IoU (%) | Precision (%) | Recall (%) |
|-------|---------|---------------|------------|
| Background | 待填写 | 待填写 | 待填写 |
| Building | 待填写 | 待填写 | 待填写 |
| Road | 待填写 | 待填写 | 待填写 |
| Water | 待填写 | 待填写 | 待填写 |
| Barren | 待填写 | 待填写 | 待填写 |
| Forest | 待填写 | 待填写 | 待填写 |
| Agricultural | 待填写 | 待填写 | 待填写 |
| **Mean** | 待填写 | 待填写 | 待填写 |

### 训练曲线

- [ ] 训练损失曲线已保存
- [ ] 验证mIoU曲线已保存
- [ ] 学习率变化曲线已保存

保存路径: `./runs/experiment_1_baseline/`

### 可视化结果

- [ ] 预测样本可视化已生成 (至少10个样本)
- [ ] 混淆矩阵已生成
- [ ] 类别分布图已生成

保存路径: `./docs/experiments/exp1_visualizations/`

## 分析

### 训练收敛性

**问题**:
- 训练损失是否收敛？
- 验证mIoU是否稳定？
- 是否存在过拟合？

**回答**: 待填写

### 性能分析

**问题**:
- 哪些类别性能较好？哪些较差？
- 性能差异的可能原因？
- 与预期性能对比如何？

**回答**: 待填写

### 存在的问题

1. 待填写
2. 待填写
3. 待填写

## 结论

### 主要发现

1. 待填写
2. 待填写
3. 待填写

### 基线性能评估

- [ ] 满足预期 (mIoU > 40%)
- [ ] 基本满足 (mIoU 30-40%)
- [ ] 不满足预期 (mIoU < 30%)

**评估**: 待填写

## 下一步计划

### 改进方向

1. **模型架构改进**:
   - [ ] 增加LoRA rank
   - [ ] 调整prompt数量
   - [ ] 尝试不同SAM backbone

2. **训练策略优化**:
   - [ ] 调整学习率schedule
   - [ ] 增加训练epochs
   - [ ] 尝试不同optimizer

3. **数据增强**:
   - [ ] 添加随机旋转/翻转
   - [ ] 添加颜色抖动
   - [ ] 添加mixup/cutmix

### 下一个实验

**实验2**: 待规划

**预期改进**: 待填写

## 附录

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

### 环境信息

- **操作系统**: macOS / Linux / Windows
- **Python版本**: 待填写
- **PyTorch版本**: 待填写
- **硬件**: 待填写
- **显存/内存**: 待填写

### 文件清单

- [ ] 训练脚本: `scripts/train.py`
- [ ] 配置文件: `config/default.py`
- [ ] 最佳模型: `checkpoints/best_model.pth`
- [ ] TensorBoard日志: `runs/experiment_1_baseline/`
- [ ] 实验记录: `docs/experiments/exp1_baseline.md` (本文件)

### 参考

- LoveDA Dataset: https://zenodo.org/record/5706578
- SAM Paper: https://arxiv.org/abs/2304.02643
- LoRA Paper: https://arxiv.org/abs/2106.09685
