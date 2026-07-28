"""
损失函数模块

实现分割任务的损失函数：
- DiceLoss: 衡量预测和真实标签的重叠程度
- FocalLoss: 处理类别不平衡问题
- CombinedLoss: 组合Dice Loss和Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss用于分割任务

    Dice系数衡量预测和真实标签的重叠程度。
    Dice Loss = 1 - Dice Coefficient

    Args:
        smooth: 平滑因子，避免除零错误，默认1.0
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets, validMask=None):
        """
        计算Dice Loss

        Args:
            preds: 预测值 (B, 1, H, W)，logits或概率
            targets: 真实标签 (B, 1, H, W)，二值标签 {0, 1}
            validMask: 有效像素掩码 (B, 1, H, W)，True表示参与计算。
                None表示全部像素有效。用于排除padding的ignore区域。

        Returns:
            loss: 标量损失值
        """
        # 将预测值转换为概率
        preds = torch.sigmoid(preds)

        targets = targets.float()

        # 屏蔽无效像素：置零后不贡献交集，也不贡献并集
        if validMask is not None:
            validMask = validMask.to(preds.dtype)
            preds = preds * validMask
            targets = targets * validMask

        # 展平张量以便计算
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)

        # 计算交集和并集
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()

        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss = 1 - Dice Coefficient
        loss = 1.0 - dice

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡问题

    降低易分类样本的权重，让模型更关注难分类样本。

    Args:
        alpha: 类别权重，默认0.25
        gamma: 聚焦参数，默认2.0，gamma越大对易分类样本的抑制越强
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets, validMask=None):
        """
        计算Focal Loss

        Args:
            preds: 预测logits (B, 1, H, W)
            targets: 真实标签 (B, 1, H, W)，二值标签 {0, 1}
            validMask: 有效像素掩码 (B, 1, H, W)，True表示参与计算。
                None表示全部像素有效。用于排除padding的ignore区域。

        Returns:
            loss: 标量损失值
        """
        # 确保targets是float类型
        targets = targets.float()

        # 使用binary_cross_entropy_with_logits计算BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            preds, targets, reduction='none'
        )

        # 计算预测概率
        probs = torch.sigmoid(preds)

        # 计算pt：正确类别的预测概率
        pt = targets * probs + (1 - targets) * (1 - probs)

        # 计算Focal Loss的调制因子
        focal_weight = (1 - pt) ** self.gamma

        # 应用alpha权重
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # 最终的Focal Loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        # 仅在有效像素上取平均
        if validMask is not None:
            validMask = validMask.to(focal_loss.dtype)
            denom = validMask.sum()
            if denom == 0:
                return focal_loss.sum() * 0.0
            return (focal_loss * validMask).sum() / denom

        # 返回平均损失
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    组合Dice Loss和Focal Loss

    Args:
        dice_weight: Dice Loss的权重，默认1.0
        focal_weight: Focal Loss的权重，默认0.5
    """

    def __init__(self, dice_weight=1.0, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, preds, targets, validMask=None):
        """
        计算组合损失

        Args:
            preds: 预测logits (B, 1, H, W)
            targets: 真实标签 (B, 1, H, W)，二值标签 {0, 1}
            validMask: 有效像素掩码 (B, 1, H, W)，True表示参与计算。
                None表示全部像素有效。

        Returns:
            loss: 标量损失值
        """
        # 计算各项损失
        dice = self.dice_loss(preds, targets, validMask)
        focal = self.focal_loss(preds, targets, validMask)

        # 加权组合
        combined = self.dice_weight * dice + self.focal_weight * focal

        return combined
