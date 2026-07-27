import pytest
import torch
from models.samLora import SAMLoRA
from models.promptLearner import SimplePromptLearner
from models.loss import DiceLoss, FocalLoss, CombinedLoss


def test_sam_lora_initialization():
    """测试SAM+LoRA模型初始化"""
    model = SAMLoRA(
        samCheckpoint=None,  # 测试模式：不加载checkpoint
        loraRank=4,
        loraAlpha=8
    )

    # 检查模型是否正确加载
    assert model.sam is not None
    assert model.loraRank == 4

    # 检查LoRA层是否正确插入
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n.lower()]
    assert len(lora_params) > 0, "LoRA参数未找到"

    # 检查SAM主干是否冻结
    sam_backbone_params = [
        p for n, p in model.named_parameters()
        if 'lora' not in n.lower() and p.requires_grad
    ]
    # 应该只有LoRA参数可训练
    assert len(sam_backbone_params) == 0, f"发现{len(sam_backbone_params)}个未冻结的非LoRA参数"


def test_sam_lora_forward():
    """测试SAM+LoRA前向传播"""
    model = SAMLoRA(loraRank=4)
    model.eval()

    # 模拟输入
    batchSize = 2
    image = torch.randn(batchSize, 3, 1024, 1024)
    prompts = torch.randn(batchSize, 256)  # 简单提示嵌入

    # 前向传播
    with torch.no_grad():
        masks = model(image, prompts)

    # 检查输出
    assert masks.shape[0] == batchSize, f"批次大小错误: {masks.shape[0]} != {batchSize}"
    assert masks.shape[1] == 1, f"掩码通道数错误: {masks.shape[1]} != 1"
    assert masks.dtype == torch.float32, f"掩码类型错误: {masks.dtype}"


def test_sam_lora_parameter_counting():
    """测试参数统计功能"""
    model = SAMLoRA(loraRank=4)

    param_stats = model.count_params()

    # 检查返回字段
    assert "total" in param_stats
    assert "trainable" in param_stats
    assert "frozen" in param_stats
    assert "trainable_ratio" in param_stats

    # 检查参数量合理性
    assert param_stats["total"] > 0
    assert param_stats["trainable"] > 0
    assert param_stats["trainable"] < param_stats["total"]
    assert param_stats["frozen"] == param_stats["total"] - param_stats["trainable"]

    # LoRA应该大幅减少训练参数（通常<1%）
    trainable_ratio = param_stats["trainable"] / param_stats["total"]
    assert trainable_ratio < 0.05, f"LoRA训练参数比例过高: {trainable_ratio:.2%}"


def test_sam_lora_different_ranks():
    """测试不同LoRA秩的模型"""
    ranks = [2, 4, 8]
    prev_trainable = None

    for rank in ranks:
        model = SAMLoRA(loraRank=rank)
        param_stats = model.count_params()

        if prev_trainable is not None:
            # 更高的秩应该有更多可训练参数
            assert param_stats["trainable"] > prev_trainable, \
                f"秩{rank}的参数量({param_stats['trainable']})应大于秩{rank//2}的参数量({prev_trainable})"

        prev_trainable = param_stats["trainable"]


def test_simple_prompt_learner_initialization():
    """测试SimplePromptLearner初始化"""
    nClasses = 5
    nPrompts = 10
    embedDim = 256

    learner = SimplePromptLearner(
        nClasses=nClasses,
        nPrompts=nPrompts,
        embedDim=embedDim,
        initStd=0.02
    )

    # 检查提示嵌入形状
    assert hasattr(learner, 'promptEmbeds'), "缺少promptEmbeds属性"
    assert learner.promptEmbeds.shape == (nClasses, nPrompts, embedDim), \
        f"提示嵌入形状错误: {learner.promptEmbeds.shape} != ({nClasses}, {nPrompts}, {embedDim})"

    # 检查参数可训练性
    assert learner.promptEmbeds.requires_grad, "提示嵌入应该是可训练的"

    # 检查初始化标准差
    std = learner.promptEmbeds.std().item()
    assert 0.01 < std < 0.05, f"初始化标准差异常: {std:.4f}"


def test_simple_prompt_learner_forward():
    """测试SimplePromptLearner前向传播"""
    nClasses = 5
    nPrompts = 10
    embedDim = 256

    learner = SimplePromptLearner(nClasses=nClasses, nPrompts=nPrompts, embedDim=embedDim)

    # 测试单个类别
    classIds = torch.tensor([0])
    prompts = learner(classIds)
    assert prompts.shape == (1, nPrompts, embedDim), \
        f"输出形状错误: {prompts.shape} != (1, {nPrompts}, {embedDim})"

    # 测试批次
    batchSize = 8
    classIds = torch.randint(0, nClasses, (batchSize,))
    prompts = learner(classIds)
    assert prompts.shape == (batchSize, nPrompts, embedDim), \
        f"批次输出形状错误: {prompts.shape} != ({batchSize}, {nPrompts}, {embedDim})"


def test_simple_prompt_learner_different_classes():
    """验证不同类别得到不同提示"""
    nClasses = 5
    nPrompts = 10
    embedDim = 256

    learner = SimplePromptLearner(nClasses=nClasses, nPrompts=nPrompts, embedDim=embedDim)

    # 获取两个不同类别的提示
    classIds1 = torch.tensor([0])
    classIds2 = torch.tensor([1])

    prompts1 = learner(classIds1)
    prompts2 = learner(classIds2)

    # 不同类别应该得到不同提示
    assert not torch.allclose(prompts1, prompts2), "不同类别的提示不应该相同"


def test_simple_prompt_learner_same_class():
    """验证相同类别得到相同提示"""
    nClasses = 5
    nPrompts = 10
    embedDim = 256

    learner = SimplePromptLearner(nClasses=nClasses, nPrompts=nPrompts, embedDim=embedDim)

    # 在批次中重复相同类别
    classIds = torch.tensor([2, 2, 2])
    prompts = learner(classIds)

    # 相同类别应该得到相同提示
    assert torch.allclose(prompts[0], prompts[1]), "相同类别的提示应该相同"
    assert torch.allclose(prompts[1], prompts[2]), "相同类别的提示应该相同"


def test_simple_prompt_learner_get_prompt_for_class():
    """测试获取单个类别提示的方法"""
    nClasses = 5
    nPrompts = 10
    embedDim = 256

    learner = SimplePromptLearner(nClasses=nClasses, nPrompts=nPrompts, embedDim=embedDim)

    # 获取单个类别提示
    classId = 3
    prompt = learner.get_prompt_for_class(classId)

    assert prompt.shape == (nPrompts, embedDim), \
        f"单类别提示形状错误: {prompt.shape} != ({nPrompts}, {embedDim})"

    # 验证与forward结果一致
    classIds = torch.tensor([classId])
    prompt_from_forward = learner(classIds)[0]

    assert torch.allclose(prompt, prompt_from_forward), \
        "get_prompt_for_class与forward结果应该一致"


def test_dice_loss():
    """测试Dice Loss"""
    loss_fn = DiceLoss(smooth=1.0)

    # 模拟预测和标签
    batchSize = 4
    height, width = 256, 256
    preds = torch.randn(batchSize, 1, height, width, requires_grad=True)
    targets = torch.randint(0, 2, (batchSize, 1, height, width)).float()

    # 计算损失
    loss = loss_fn(preds, targets)

    # 检查输出
    assert loss.shape == torch.Size([]), f"损失应该是标量，实际形状: {loss.shape}"
    assert loss.item() >= 0, f"Dice Loss应该非负，实际值: {loss.item()}"
    assert loss.requires_grad, "损失应该支持梯度"

    # 测试梯度反传
    loss.backward()
    assert preds.grad is not None, "预测值应该有梯度"
    assert preds.grad.shape == preds.shape, "梯度形状应该与输入相同"


def test_dice_loss_perfect_match():
    """测试Dice Loss在完美匹配时接近0"""
    loss_fn = DiceLoss(smooth=1.0)

    # 完美匹配的情况
    targets = torch.ones(2, 1, 64, 64)
    preds = torch.ones(2, 1, 64, 64) * 10.0  # 经过sigmoid后接近1

    loss = loss_fn(preds, targets)

    # 完美匹配时损失应该接近0
    assert loss.item() < 0.1, f"完美匹配时损失应接近0，实际值: {loss.item()}"


def test_focal_loss():
    """测试Focal Loss"""
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # 模拟预测logits和标签
    batchSize = 4
    height, width = 256, 256
    preds = torch.randn(batchSize, 1, height, width, requires_grad=True)
    targets = torch.randint(0, 2, (batchSize, 1, height, width)).float()

    # 计算损失
    loss = loss_fn(preds, targets)

    # 检查输出
    assert loss.shape == torch.Size([]), f"损失应该是标量，实际形状: {loss.shape}"
    assert loss.item() >= 0, f"Focal Loss应该非负，实际值: {loss.item()}"
    assert loss.requires_grad, "损失应该支持梯度"

    # 测试梯度反传
    loss.backward()
    assert preds.grad is not None, "预测值应该有梯度"
    assert preds.grad.shape == preds.shape, "梯度形状应该与输入相同"


def test_focal_loss_easy_vs_hard():
    """测试Focal Loss对易分类样本的权重降低"""
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # 易分类样本：预测与标签一致且置信度高
    easy_preds = torch.ones(2, 1, 64, 64) * 5.0  # 高置信度预测为1
    easy_targets = torch.ones(2, 1, 64, 64)

    # 难分类样本：预测与标签不一致
    hard_preds = torch.ones(2, 1, 64, 64) * -5.0  # 高置信度预测为0
    hard_targets = torch.ones(2, 1, 64, 64)  # 但标签是1

    easy_loss = loss_fn(easy_preds, easy_targets)
    hard_loss = loss_fn(hard_preds, hard_targets)

    # 难分类样本的损失应该明显大于易分类样本
    assert hard_loss.item() > easy_loss.item() * 5, \
        f"难分类样本损失({hard_loss.item():.4f})应远大于易分类样本({easy_loss.item():.4f})"


def test_combined_loss():
    """测试Combined Loss"""
    loss_fn = CombinedLoss(dice_weight=1.0, focal_weight=0.5)

    # 模拟预测logits和标签
    batchSize = 4
    height, width = 256, 256
    preds = torch.randn(batchSize, 1, height, width, requires_grad=True)
    targets = torch.randint(0, 2, (batchSize, 1, height, width)).float()

    # 计算损失
    loss = loss_fn(preds, targets)

    # 检查输出
    assert loss.shape == torch.Size([]), f"损失应该是标量，实际形状: {loss.shape}"
    assert loss.item() >= 0, f"Combined Loss应该非负，实际值: {loss.item()}"
    assert loss.requires_grad, "损失应该支持梯度"

    # 测试梯度反传
    loss.backward()
    assert preds.grad is not None, "预测值应该有梯度"
    assert preds.grad.shape == preds.shape, "梯度形状应该与输入相同"


def test_combined_loss_weights():
    """测试Combined Loss权重设置"""
    # 纯Dice Loss
    dice_only = CombinedLoss(dice_weight=1.0, focal_weight=0.0)
    # 纯Focal Loss
    focal_only = CombinedLoss(dice_weight=0.0, focal_weight=1.0)
    # 组合Loss
    combined = CombinedLoss(dice_weight=1.0, focal_weight=1.0)

    # 相同输入
    preds = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss_dice = dice_only(preds.clone().detach().requires_grad_(True), targets)
    loss_focal = focal_only(preds.clone().detach().requires_grad_(True), targets)
    loss_combined = combined(preds.clone().detach().requires_grad_(True), targets)

    # 组合损失应该大于任一单独损失（权重为1.0时）
    assert loss_combined.item() > loss_dice.item() * 0.5, \
        "组合损失应该反映Dice Loss的贡献"
    assert loss_combined.item() > loss_focal.item() * 0.5, \
        "组合损失应该反映Focal Loss的贡献"
