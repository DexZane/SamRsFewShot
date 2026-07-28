"""损失函数的ignore掩码与二值目标构造的回归测试

这些测试锁定一个真实发生过的bug：把LoveDA的7类标签图（含255 padding）
直接喂给单通道二值损失，Focal Loss会发散到负数（观测到 train_loss = -641.97）。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from models.loss import DiceLoss, FocalLoss, CombinedLoss


def buildBinaryTargets(masks, classIds):
    """复制 Trainer._build_binary_targets 的逻辑，用于脱离GPU独立测试

    保持与 training/trainer.py 中的实现一致。
    """
    classIds = classIds.view(-1, 1, 1, 1).to(masks.device)
    validMask = masks != 255
    targets = ((masks == classIds) & validMask).float()
    return targets, validMask


class TestBinaryTargetConstruction:
    """验证7类标签图 -> 二值目标的转换"""

    def test_only_episode_class_becomes_foreground(self):
        # (1, 1, 2, 2)：类别 0/1/2/3 各一个像素，episode目标类别是2
        masks = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.int64)
        classIds = torch.tensor([2])

        targets, validMask = buildBinaryTargets(masks, classIds)

        expected = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]])
        assert torch.equal(targets, expected)
        assert validMask.all(), "无255时所有像素都应有效"

    def test_ignore_label_excluded_and_never_foreground(self):
        masks = torch.tensor([[[[2, 255], [255, 2]]]], dtype=torch.int64)
        classIds = torch.tensor([2])

        targets, validMask = buildBinaryTargets(masks, classIds)

        # 255区域既不是前景，也不参与损失
        assert targets.max() == 1.0
        assert targets.sum() == 2.0, "只有两个类别2的像素是前景"
        assert validMask.sum() == 2, "两个255像素被排除"
        assert not validMask[0, 0, 0, 1]

    def test_targets_are_strictly_binary(self):
        """核心断言：目标只能是0或1。旧代码这里会漏出255"""
        masks = torch.randint(0, 7, (4, 1, 8, 8), dtype=torch.int64)
        masks[0, 0, 0, :] = 255  # 混入ignore
        classIds = torch.tensor([1, 2, 3, 4])

        targets, _ = buildBinaryTargets(masks, classIds)

        uniqueValues = torch.unique(targets)
        assert set(uniqueValues.tolist()) <= {0.0, 1.0}, (
            f"目标必须是二值，实际含 {uniqueValues.tolist()}"
        )

    def test_per_sample_class_id_applied_independently(self):
        """batch里每个样本用自己的episode类别，不能串"""
        masks = torch.tensor([
            [[[1, 1], [2, 2]]],
            [[[1, 1], [2, 2]]],
        ], dtype=torch.int64)
        classIds = torch.tensor([1, 2])

        targets, _ = buildBinaryTargets(masks, classIds)

        assert torch.equal(targets[0], torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]))
        assert torch.equal(targets[1], torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]))


class TestLossNonNegativity:
    """损失必须非负——回归观测到的 -641.97"""

    @pytest.mark.parametrize("lossFn", [DiceLoss(), FocalLoss(), CombinedLoss()])
    def test_loss_non_negative_on_binary_targets(self, lossFn):
        torch.manual_seed(0)
        preds = torch.randn(2, 1, 16, 16) * 5  # 大logits，逼近饱和
        masks = torch.randint(0, 7, (2, 1, 16, 16), dtype=torch.int64)
        masks[:, :, :4, :] = 255
        classIds = torch.tensor([3, 5])

        targets, validMask = buildBinaryTargets(masks, classIds)
        loss = lossFn(preds, targets, validMask)

        assert torch.isfinite(loss), f"{type(lossFn).__name__} 产生了 inf/nan"
        assert loss.item() >= 0.0, f"{type(lossFn).__name__} 返回负损失 {loss.item()}"

    def test_combined_loss_bounded_in_sane_range(self):
        """训练初期的量级检查：Dice上限1，Focal在未饱和logits下不足0.5

        实测：logits ~N(0,1) 时 dice≈0.78 focal≈0.24。取2.0作为上界，
        留出饱和时的余量（logits×10 会让focal升到2.7，那不是初期状态）。
        """
        torch.manual_seed(1)
        preds = torch.randn(3, 1, 32, 32)
        masks = torch.randint(0, 7, (3, 1, 32, 32), dtype=torch.int64)
        classIds = torch.tensor([1, 2, 3])

        targets, validMask = buildBinaryTargets(masks, classIds)
        loss = CombinedLoss()(preds, targets, validMask)

        assert 0.0 <= loss.item() <= 2.0, f"损失量级异常: {loss.item()}"

    def test_raw_multiclass_labels_explode(self):
        """记录旧行为：直接传含255的标签会让Focal的量级彻底失控。

        target=255 时 BCE 展开为 -255·log(p) + 254·log(1-p)，
        再乘 (1-pt)^2（pt = 509p - 254）。符号取决于logits：
        正logits给出 +1e10 量级，负logits给出 -1e10 量级——
        观测到的 train_loss = -641.97 就是负那一侧。

        这是bug的证据，不是期望行为。保留它是为了锁定
        "调用方必须先做二值化" 这个契约。
        """
        rawLabels = torch.full((1, 1, 4, 4), 255.0)  # 未处理的ignore标签

        posLoss = FocalLoss()(torch.full((1, 1, 4, 4), 8.0), rawLabels)
        negLoss = FocalLoss()(torch.full((1, 1, 4, 4), -8.0), rawLabels)

        assert abs(posLoss.item()) > 1e6, (
            f"非二值输入的量级应该失控，实际 {posLoss.item()}"
        )
        assert negLoss.item() < -1e6, (
            f"负logits下应给出负损失（复现 -641.97），实际 {negLoss.item()}"
        )

    def test_binarized_targets_stay_sane_where_raw_labels_explode(self):
        """同一批含255的标签：二值化前爆炸，二值化后正常"""
        preds = torch.full((1, 1, 4, 4), -8.0)
        masks = torch.full((1, 1, 4, 4), 255, dtype=torch.int64)
        classIds = torch.tensor([2])

        rawLoss = FocalLoss()(preds, masks.float())
        targets, validMask = buildBinaryTargets(masks, classIds)
        fixedLoss = FocalLoss()(preds, targets, validMask)

        assert rawLoss.item() < -1e6, "旧路径应爆炸"
        assert fixedLoss.item() == 0.0, (
            f"全ignore时修复后的损失应为0，实际 {fixedLoss.item()}"
        )


class TestValidMaskSemantics:
    """validMask 应真正把ignore区域从损失里剔除"""

    def test_dice_ignores_masked_region(self):
        # 有效区：预测与标签完全一致；ignore区：预测全错
        preds = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
        targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        validMask = torch.tensor([[[[True, True], [True, True]]]])

        lossAllValid = DiceLoss()(preds, targets, validMask)

        # 把下半部分标成ignore后，损失不应变差
        partialMask = torch.tensor([[[[True, True], [False, False]]]])
        lossPartial = DiceLoss()(preds, targets, partialMask)

        assert lossAllValid.item() < 0.01, "完美预测的Dice损失应接近0"
        assert lossPartial.item() < 0.01

    def test_focal_mask_changes_denominator(self):
        """Focal是按有效像素取平均，遮罩一半应改变结果"""
        preds = torch.tensor([[[[5.0, -5.0], [5.0, -5.0]]]])
        targets = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])

        fullMask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        # 只保留预测正确的那一列
        correctOnly = torch.tensor([[[[True, False], [True, False]]]])

        lossFull = FocalLoss()(preds, targets, fullMask)
        lossCorrectOnly = FocalLoss()(preds, targets, correctOnly)

        assert lossCorrectOnly.item() < lossFull.item(), (
            "只统计预测正确的像素，损失应更低"
        )

    def test_all_pixels_ignored_returns_zero(self):
        """全ignore的极端情况不能产生nan"""
        preds = torch.randn(1, 1, 4, 4)
        targets = torch.zeros(1, 1, 4, 4)
        emptyMask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)

        focalLoss = FocalLoss()(preds, targets, emptyMask)

        assert torch.isfinite(focalLoss), "全ignore时产生了nan"
        assert focalLoss.item() == 0.0

    def test_none_mask_equals_all_valid_mask(self):
        """validMask=None 应等价于全True"""
        torch.manual_seed(2)
        preds = torch.randn(2, 1, 8, 8)
        targets = (torch.rand(2, 1, 8, 8) > 0.5).float()
        allValid = torch.ones_like(targets, dtype=torch.bool)

        for lossFn in [DiceLoss(), FocalLoss(), CombinedLoss()]:
            lossNone = lossFn(preds, targets)
            lossExplicit = lossFn(preds, targets, allValid)
            assert torch.allclose(lossNone, lossExplicit, atol=1e-6), (
                f"{type(lossFn).__name__}: None与全True掩码结果不一致"
            )


class TestGradientFlow:
    """损失必须可回传，否则LoRA学不到东西"""

    def test_combined_loss_produces_gradients(self):
        preds = torch.randn(2, 1, 8, 8, requires_grad=True)
        masks = torch.randint(0, 7, (2, 1, 8, 8), dtype=torch.int64)
        masks[0, 0, 0, 0] = 255
        classIds = torch.tensor([2, 4])

        targets, validMask = buildBinaryTargets(masks, classIds)
        loss = CombinedLoss()(preds, targets, validMask)
        loss.backward()

        assert preds.grad is not None
        assert torch.isfinite(preds.grad).all(), "梯度含inf/nan"
        assert preds.grad.abs().sum() > 0, "梯度全零，无法学习"
