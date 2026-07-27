"""
Simple Prompt Learner for Few-shot Learning

为每个类别学习可训练的提示嵌入向量，用于条件化SAM模型。
"""

import torch
import torch.nn as nn


class SimplePromptLearner(nn.Module):
    """
    简单提示学习器

    为每个类别维护一组可学习的提示嵌入向量。
    每个类别有固定数量的提示，形状为(nPrompts, embedDim)。

    Args:
        nClasses: 类别数量
        nPrompts: 每个类别的提示数量
        embedDim: 提示嵌入维度
        initStd: 初始化标准差（默认0.02）
    """

    def __init__(self, nClasses, nPrompts, embedDim, initStd=0.02):
        super().__init__()

        self.nClasses = nClasses
        self.nPrompts = nPrompts
        self.embedDim = embedDim

        # 初始化可学习的提示嵌入
        # Shape: (nClasses, nPrompts, embedDim)
        self.promptEmbeds = nn.Parameter(
            torch.randn(nClasses, nPrompts, embedDim) * initStd
        )

    def forward(self, classIds):
        """
        根据类别ID返回对应的提示嵌入

        Args:
            classIds: 类别ID张量，shape (batchSize,)

        Returns:
            prompts: 提示嵌入张量，shape (batchSize, nPrompts, embedDim)
        """
        # 通过索引获取对应类别的提示
        prompts = self.promptEmbeds[classIds]
        return prompts

    def get_prompt_for_class(self, classId):
        """
        获取单个类别的提示嵌入（用于分析）

        Args:
            classId: 类别ID（整数）

        Returns:
            prompt: 该类别的提示嵌入，shape (nPrompts, embedDim)
        """
        return self.promptEmbeds[classId]
