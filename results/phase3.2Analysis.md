# Phase 3.2 训练分析报告

## 实验配置

**Phase 3.2 改进点：增加模型容量**
- LoRA rank: 4 → **8** (翻倍)
- LoRA alpha: 8 → **16** (翻倍)
- nPrompts: 5 → **10** (翻倍)
- Dropout: 0.2 (保持不变)
- 数据增强: 启用 (与Phase 3.1相同)

**假设**：Phase 3.1训练loss高企(0.6155)是因为模型容量不足，增大容量后应能更好地学习增强数据

## 训练结果

### 关键指标

| 指标 | Phase 2 (baseline) | Phase 3.1 (aug+dropout) | Phase 3.2 (aug+capacity) | 变化 |
|------|-------------------|------------------------|--------------------------|------|
| **最佳 mIoU** | 0.5040 | 0.5065 | **0.5070** | +0.0005 (+0.1%) |
| **最佳 Epoch** | 10 | 45 | **10** | 回退到早期 |
| **训练 Loss (最后10 epoch)** | 0.2442 | 0.6155 | **0.6308** | 更差 |
| **早停 Epoch** | 60 | 95 | **60** | 提前触发 |
| **训练稳定性** | 严重过拟合 | 欠拟合 | **严重欠拟合** | 恶化 |

### 训练Loss曲线

**前10个epoch的训练loss：**
```
Epoch 1:  0.7354
Epoch 2:  0.7373
Epoch 3:  0.7038
Epoch 4:  0.6776
Epoch 5:  0.6572
Epoch 6:  0.6734
Epoch 7:  0.6261
Epoch 8:  0.5961
Epoch 9:  0.6856
Epoch 10: 0.7266
```

**观察：**
- Loss下降极慢，前10个epoch仅从0.735降至0.596
- Epoch 10后loss开始**上升**至0.726（训练不稳定）
- 最后10个epoch平均loss=0.6308，远高于Phase 2的0.2442

### 验证集mIoU曲线

```
Epoch 5:  0.4896
Epoch 10: 0.5070 ← 最佳
Epoch 15: 0.4868 ↓
Epoch 20: 0.4974
Epoch 25: 0.4803 ↓
Epoch 30: 0.4956
Epoch 35: 0.4956
Epoch 40: 0.5013
Epoch 45: 0.5007
Epoch 50: 0.4796 ↓
Epoch 55: 0.4901
Epoch 60: 0.4885 (触发早停)
```

**观察：**
- 最佳mIoU在Epoch 10达到，与Phase 2相同
- 此后50个epoch无任何改进，波动范围0.48-0.50
- mIoU曲线极不稳定，没有明确的上升或下降趋势

## 核心问题诊断

### ❌ Phase 3.2 **失败** - 容量假设不成立

**预期：** 增加模型容量后，训练loss应降至0.35-0.40，mIoU提升至0.54+

**实际：** 训练loss反而更高(0.6308 vs 0.6155)，mIoU几乎无变化(+0.0005)

### 根本原因分析

1. **数据增强过强导致任务难度爆炸**
   - 训练loss高企(0.6+)说明模型根本无法拟合训练集
   - 增加容量无效，说明问题不在模型，在数据

2. **Few-shot学习 + 强增强 = 灾难性组合**
   - Few-shot本就样本极少(每类2张support)
   - 强增强(翻转+旋转+颜色扰动)破坏了support-query的相似性
   - 模型无法从2张变化剧烈的support图像中提取稳定特征

3. **训练不稳定**
   - Loss曲线震荡(Epoch 8: 0.596 → Epoch 10: 0.726)
   - mIoU曲线无规律波动
   - 说明优化过程陷入混乱

## 三阶段对比总结

| 阶段 | 策略 | 训练Loss | 验证mIoU | 结论 |
|------|------|---------|---------|------|
| Phase 2 | 无增强 | 0.24 (过拟合) | 0.5040 | 基线 |
| Phase 3.1 | 增强+dropout | 0.62 (欠拟合) | 0.5065 | 增强过强 |
| Phase 3.2 | 增强+容量↑ | 0.63 (欠拟合) | 0.5070 | 容量无效 |

**结论：当前的数据增强策略不适合few-shot场景**

## 下一步改进方向

### 方向A：温和数据增强 ⭐ 推荐

**策略：**
1. 仅保留几何增强(翻转+小角度旋转±15°)
2. 移除颜色扰动(ColorJitter)
3. 降低dropout至0.1
4. 保持较小的模型容量(LoRA rank=4, prompts=5)

**理由：**
- 几何增强不改变语义信息
- 颜色扰动可能破坏遥感影像的光谱特征
- Few-shot需要温和增强来保持特征一致性

**预期：**
- 训练loss降至0.35-0.45
- 验证mIoU提升至0.52-0.54

### 方向B：增加训练数据量

**策略：**
1. 增加support shot数: 2-shot → 5-shot
2. 增加训练episodes: 50 → 100
3. 保持温和增强

**理由：**
- 更多support样本提供更稳定的类别表征
- 更多episodes增加模型见过的变化模式

**预期：**
- 提升泛化能力
- 可能需要更长训练时间

### 方向C：改进训练策略

**策略：**
1. 使用余弦退火学习率 (CosineAnnealingLR)
2. Warmup前5个epoch
3. 梯度裁剪防止训练不稳定

**理由：**
- 当前StepLR可能导致学习率下降过快
- Warmup有助于模型在增强数据上稳定收敛

## 建议行动

**立即执行：方向A (温和增强)**

修改 `data/augmentedTransform.py`:
```python
class MildAugmentedTransform:
    def __init__(self, target_size=1024, train=True, 
                 hflip_prob=0.5, vflip_prob=0.5, 
                 rotation_prob=0.3, max_rotation=15):  # ← 关键改动
        self.train = train
        self.target_size = target_size
        if train:
            self.hflip_prob = hflip_prob
            self.vflip_prob = vflip_prob
            self.rotation_prob = rotation_prob
            self.max_rotation = max_rotation
            # 移除 ColorJitter ← 关键改动
```

修改 `config/default.py`:
```python
loraRank: int = 4       # 恢复小容量
loraAlpha: int = 8
loraDropout: float = 0.1  # 降低dropout
nPrompts: int = 5       # 恢复少量prompts
```

**预期收益：**
- 训练更稳定
- Loss能正常下降
- mIoU有望突破0.52

## 实验记录

- **启动时间**: 2026-07-29 15:55
- **结束时间**: 2026-07-29 17:04
- **总耗时**: ~69分钟
- **早停Epoch**: 60
- **Checkpoint**: `best_model_phase3.2.pth` (Epoch 10)
- **可视化结果**: `results/phase3.2_predictions/`

## 失败教训

1. **不要盲目增加模型容量** - 容量不是万能药
2. **Few-shot场景需要温和增强** - 强增强破坏特征一致性
3. **训练loss是关键诊断指标** - Loss高企说明任务定义有问题
4. **早期收敛不一定是坏事** - Phase 2虽然Epoch 10就收敛，但mIoU不差

---

**Phase 3.2状态**: ❌ 失败 - 容量增加无效，数据增强策略需要重新设计
