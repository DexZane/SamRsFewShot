# SAM遥感少样本分割 Phase 1

基于Segment Anything Model (SAM)的遥感影像少样本分割基础实现。

## Phase 1 目标

构建基础版本（v0.1）：SAM + LoRA + 简单Prompt Learning

## 安装

```bash
pip install -e .
```

## 快速开始

```bash
# 训练
python scripts/train.py --dataRoot /path/to/loveda --nWay 5 --kShot 5

# 评估
python scripts/evaluate.py --checkpoint checkpoints/best.pth
```

## 实验设置

- 数据集：LoveDA
- 少样本设置：5-way 5-shot
- 模型：SAM-ViT-B + LoRA (rank=4)
