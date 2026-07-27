import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from typing import Optional
import math


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer

    实现 W = W0 + BA，其中 B 是 (d, r)，A 是 (r, k)
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: int = 8, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算 LoRA 的增量输出"""
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LinearWithLoRA(nn.Module):
    """带LoRA的线性层"""
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: int = 8, dropout: float = 0.1):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha, dropout)

        # 冻结原始权重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


class SAMLoRA(nn.Module):
    """SAM + LoRA适配器

    在SAM的图像编码器中插入LoRA层，实现参数高效微调。
    冻结SAM主干参数，仅训练LoRA适配器。

    Args:
        samCheckpoint: SAM预训练权重路径（None表示测试模式，不加载checkpoint）
        model_type: SAM模型类型 ('vit_b', 'vit_l', 'vit_h')
        loraRank: LoRA秩（rank），控制低秩矩阵的维度
        loraAlpha: LoRA缩放因子，影响LoRA权重的更新幅度
        loraDropout: LoRA dropout率

    Examples:
        >>> # 测试模式（不加载checkpoint）
        >>> model = SAMLoRA(loraRank=4)
        >>>
        >>> # 生产模式（加载预训练权重）
        >>> model = SAMLoRA(samCheckpoint="sam_vit_b.pth", loraRank=8)
    """

    def __init__(
        self,
        samCheckpoint: Optional[str] = None,
        model_type: str = "vit_b",
        loraRank: int = 4,
        loraAlpha: int = 8,
        loraDropout: float = 0.1
    ):
        super().__init__()

        self.model_type = model_type
        self.loraRank = loraRank
        self.loraAlpha = loraAlpha

        # 加载SAM模型
        if samCheckpoint is not None:
            self.sam = sam_model_registry[model_type](checkpoint=samCheckpoint)
        else:
            # 测试模式：不加载checkpoint，避免下载大文件
            self.sam = sam_model_registry[model_type]()

        # 冻结SAM主干参数
        for param in self.sam.parameters():
            param.requires_grad = False

        # 手动注入LoRA层到注意力模块的qkv投影
        self._inject_lora_layers(loraRank, loraAlpha, loraDropout)

        # 获取embedding维度
        self.embedDim = self.sam.prompt_encoder.embed_dim

    def _inject_lora_layers(self, rank: int, alpha: int, dropout: float):
        """在图像编码器的注意力层中注入LoRA"""
        for block in self.sam.image_encoder.blocks:
            # 替换qkv线性层
            original_qkv = block.attn.qkv
            block.attn.qkv = LinearWithLoRA(original_qkv, rank, alpha, dropout)

    def forward(self, images: torch.Tensor, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            images: 输入图像 (B, 3, 1024, 1024)
            prompt_embeddings: 提示嵌入 (B, embedDim)

        Returns:
            masks: 预测掩码 (B, 1, H, W)
        """
        # 图像编码（通过带LoRA的encoder）
        image_embeddings = self.sam.image_encoder(images)

        # 准备提示嵌入
        batchSize = images.shape[0]
        sparse_embeddings = prompt_embeddings.unsqueeze(1)  # (B, 1, embedDim)

        # 创建空的dense embeddings
        dense_embeddings = torch.zeros(
            batchSize,
            self.embedDim,
            image_embeddings.shape[-2],
            image_embeddings.shape[-1],
            device=images.device,
            dtype=images.dtype
        )

        # 逐样本处理掩码解码（避免batch维度问题）
        masks_list = []
        for i in range(batchSize):
            low_res_mask, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings[i:i+1],
                dense_prompt_embeddings=dense_embeddings[i:i+1],
                multimask_output=False
            )
            masks_list.append(low_res_mask)

        # 合并所有样本的masks
        low_res_masks = torch.cat(masks_list, dim=0)

        return low_res_masks

    def get_trainable_params(self):
        """获取可训练参数列表

        Returns:
            list: 所有requires_grad=True的参数
        """
        return [p for p in self.parameters() if p.requires_grad]

    def count_params(self):
        """统计参数量

        Returns:
            dict: 包含total, trainable, frozen, trainable_ratio的字典
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.get_trainable_params())

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_ratio": f"{100 * trainable / total:.2f}%"
        }
