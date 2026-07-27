# SAM-RS Few-Shot Project Index

**项目全称**: SAM-RS Few-Shot Semantic Segmentation  
**当前阶段**: Phase 2 - 基线训练实验  
**最后更新**: 2026-07-27

---

## 📋 快速导航

### 核心文档
- [Phase 2 设置指南](../PHASE2_SETUP.md) - 如何启动训练
- [Phase 2 状态报告](Phase2StatusReport.md) - 当前进度详情
- [行动清单](ActionItems.md) - 待办事项和优先级
- [Phase 1 总结](../PHASE1_SUMMARY.md) - 数据加载器实现回顾

### 实验记录
- [实验1模板](experiments/exp1_baseline_template.md) - 基线实验记录模板

### 技术文档
- [数据集设计](superpowers/specs/2026-07-27-loveda-dataset-loader-design.md) - LoveDA加载器架构
- [实现计划](superpowers/plans/2026-07-27-loveda-dataset-loader.md) - Phase 1开发计划

---

## 🎯 项目目标

### 研究目标
利用SAM (Segment Anything Model) + LoRA + Prompt Learning实现遥感影像few-shot语义分割

### 技术创新点
1. **参数高效适配**: 使用LoRA减少可训练参数
2. **Prompt Learning**: 每类学习可学习提示向量
3. **Few-shot设置**: 5-way 5-shot语义分割

### 应用场景
- 遥感影像土地覆盖分类
- 少样本场景快速适配
- 跨域语义分割

---

## 📁 项目结构

```
samRsFewShot/
├── config/                  # 配置文件
│   └── default.py          # 默认配置
├── data/                    # 数据加载模块
│   ├── download.py         # 数据集下载
│   ├── transforms.py       # 图像预处理
│   ├── lovedaDataset.py    # LoveDA数据集类
│   └── fewShotSampler.py   # Few-shot采样器
├── models/                  # 模型定义
│   ├── samLora.py          # SAM + LoRA
│   ├── promptLearner.py    # Prompt学习器
│   └── loss.py             # 损失函数
├── training/                # 训练模块
│   └── trainer.py          # 训练器
├── scripts/                 # 执行脚本
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   ├── visualizeDataset.py # 数据可视化
│   └── run_phase2_baseline.sh  # Phase 2启动脚本
├── tests/                   # 单元测试
│   ├── test_download.py
│   ├── test_transforms.py
│   └── test_loveda_dataset.py
├── docs/                    # 文档
│   ├── IndexProject.md     # 本文件
│   ├── Phase2StatusReport.md
│   ├── ActionItems.md
│   └── experiments/        # 实验记录
├── checkpoints/            # 模型检查点
├── runs/                   # TensorBoard日志
└── data/                   # 数据目录
    └── LoveDA/             # LoveDA数据集
```

---

## 🚀 快速开始

### 环境准备
```bash
# 1. 克隆项目
cd samRsFewShot

# 2. 安装依赖
pip install -e .

# 3. 运行测试
pytest tests/ -v
```

### 启动训练
```bash
# 1. 下载数据集（手动）
# 访问: https://zenodo.org/record/5706578
# 解压到: ./data/LoveDA/

# 2. 启动训练（自动下载SAM模型）
bash scripts/run_phase2_baseline.sh
```

### 查看结果
```bash
# TensorBoard可视化
tensorboard --logdir=./runs

# 检查最佳模型
ls -lh checkpoints/best_model.pth
```

---

## 📊 当前状态

### Phase 1: 数据加载器 ✅
- **状态**: 已完成
- **完成时间**: 2026-07-27
- **产出**: 
  - 6个Python模块
  - 12个单元测试（全部通过）
  - 完整文档

### Phase 2: 基线训练 ⏸️
- **状态**: 待启动（前置准备完成30%）
- **开始时间**: 2026-07-27
- **阻塞**: 等待LoveDA数据集下载
- **下一步**: 
  1. 下载数据集
  2. 启动训练
  3. 评估结果

### Phase 3: 方法改进 ⏸️
- **状态**: 未启动
- **依赖**: Phase 2完成
- **计划**: 
  - 模型架构优化
  - 超参数调优
  - 消融实验

---

## 📈 进度追踪

### 总体进度

| 阶段 | 进度 | 状态 |
|------|------|------|
| Phase 1: 数据加载器 | 100% | ✅ 完成 |
| Phase 2: 基线训练 | 30% | ⏸️ 进行中 |
| Phase 3: 方法改进 | 0% | ⏸️ 未启动 |
| Phase 4: 论文撰写 | 0% | ⏸️ 未启动 |

### 关键里程碑

- [x] 2026-07-27: 项目启动
- [x] 2026-07-27: Phase 1完成
- [x] 2026-07-27: Phase 2准备完成
- [ ] 待定: 首次训练启动
- [ ] 待定: 基线实验完成
- [ ] 待定: Phase 2完成
- [ ] 待定: Phase 3启动

---

## 🔧 技术栈

### 核心框架
- **PyTorch**: 2.13+
- **Segment Anything (SAM)**: 1.0
- **PEFT (LoRA)**: 0.19+

### 数据处理
- **PIL**: 图像读取
- **NumPy**: 数组操作
- **torchvision**: 图像变换

### 训练工具
- **TensorBoard**: 可视化
- **tqdm**: 进度条
- **pytest**: 单元测试

### 硬件支持
- **MPS**: Apple Silicon GPU
- **CUDA**: NVIDIA GPU
- **CPU**: 通用CPU（不推荐）

---

## 📦 数据集信息

### LoveDA Dataset
- **来源**: Zenodo (DOI: 10.5281/zenodo.5706578)
- **类型**: 遥感影像土地覆盖数据集
- **类别数**: 7类
  - Background (背景)
  - Building (建筑)
  - Road (道路)
  - Water (水体)
  - Barren (裸地)
  - Forest (森林)
  - Agricultural (农田)
- **划分**: Train / Val / Test
- **场景**: Urban (城市) + Rural (乡村)
- **大小**: ~2.5 GB

### 数据统计
- **训练集**: ~2000张图像
- **验证集**: ~500张图像
- **测试集**: ~500张图像
- **图像尺寸**: 1024×1024

---

## 🎓 相关论文

### 基础模型
1. **SAM**: Kirillov et al., "Segment Anything", ICCV 2023
   - https://arxiv.org/abs/2304.02643

2. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
   - https://arxiv.org/abs/2106.09685

### 相关工作
3. **Few-shot Segmentation**: 待补充
4. **Remote Sensing Segmentation**: 待补充
5. **Prompt Learning**: 待补充

---

## 📝 实验记录

### 已完成实验
- 无（Phase 2未启动）

### 计划实验
1. **Exp 1**: Baseline (5-way 5-shot)
2. **Exp 2**: 增加LoRA rank
3. **Exp 3**: 调整prompt数量
4. **Exp 4**: 数据增强
5. **Exp 5**: 不同few-shot配置

---

## 🐛 已知问题

### 高优先级
- 无

### 中优先级
- LoveDA数据集自动下载链接失效（已提供手动下载方案）

### 低优先级
- 无

---

## 🤝 贡献指南

### 开发规范
1. 代码风格: PEP 8
2. 测试覆盖: ≥80%
3. 文档: 每个模块必需
4. Commit: 语义化commit message

### 分支策略
- `main`: 稳定版本
- `dev`: 开发版本
- `feature/*`: 功能分支
- `exp/*`: 实验分支

---

## 📞 联系方式

- **项目负责人**: 待填写
- **研究团队**: 待填写
- **GitHub**: 待填写
- **邮箱**: 待填写

---

## 📄 许可证

待定

---

## 🔗 外部链接

- [LoveDA Dataset](https://zenodo.org/record/5706578)
- [SAM GitHub](https://github.com/facebookresearch/segment-anything)
- [PEFT GitHub](https://github.com/huggingface/peft)
- [PyTorch](https://pytorch.org/)

---

**最后更新**: 2026-07-27  
**维护者**: AI Assistant
