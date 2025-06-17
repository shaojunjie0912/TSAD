"""
通道融合模块 (Channel Fusion Module, CFM) 核心实现
"""

import math
from typing import Optional, Tuple, Union

import einops
import torch
import torch.nn.functional as F
from torch import nn

from ..criterions.channel_discover_loss import AdaptiveCCDLoss


class PreNorm(nn.Module):
    """预归一化层 (层归一化)

    module 的输入是经过层归一化的, 这样可以避免在每个子层中都进行归一化操作
    NOTE: 缓解模型对高幅值频率分量的过度关注现象
    """

    def __init__(self, normalized_shape: int, module: nn.Module):
        """_summary_

        Args:
            normalized_shape (int): 输入特征维度
            fn (nn.Module): 要应用的函数(如注意力机制或前馈网络)
        """
        # TODO: 传入一个 module 作为形参有点奇怪吧...感觉是为了省事,
        # 即 module 的输入都是经过层归一化的
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=normalized_shape)  # 层归一化
        self.module = module

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # NOTE: kwargs 会传入 module 中的 forward 函数, 字典键必须对应!
        return self.module(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """前馈网络层

    数据流: 多头注意力 -> 残差连接 -> 层归一化 -> 前馈网络 -> 残差连接 -> 层归一化
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.5):
        """_summary_

        Args:
            dim (int): 输入和输出特征的维度
            hidden_dim (int): 隐藏层特征维度
            dropout (float, optional): 丢弃率, 防止过拟合, 默认 0.5.
        """
        super().__init__()
        self.ff_layer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # TODO: Dropout 层作用?
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_layer(x)


class ChannelMaskedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        d_head: int,
        dropout: float = 0.8,
        ccd_temperature: float = 0.1,  # 用于 AdaptiveCCDLoss
        ccd_regular_lambda: float = 0.1,  # 用于 AdaptiveCCDLoss
        ccd_alignment_lambda: float = 1.0,  # 新增
    ):
        """

        Args:
            dim (int): TODO: 输入之前由 patch size * 2 映射到 dim
            heads (int): 多头注意力机制的头数
            d_head (int): 每个头的特征维度 (也是 Transformer 维度?)
            dropout (float, optional): 丢弃率, 防止过拟合 (默认 0.8)
            regular_lambda (float, optional): 对比损失的正则化系数 (默认 0.3)
            temperature (float, optional): 对比损失的温度参数 (默认 0.1)
        """
        super().__init__()
        self.num_heads = num_heads  # 多头注意力中的头数量, 每个头独立计算注意力，捕获不同的特征模式
        self.d_head = d_head  # 每个注意力头的维度大小, 决定了每个头处理的特征向量的长度
        inner_dim = d_head * num_heads  # 所有头的总维度
        self.to_q = nn.Linear(dim, inner_dim)  # (..., dim) -> (..., inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # 通道关联性发掘损失函数
        self.ccd_loss_fn = AdaptiveCCDLoss(
            alignment_temperature=ccd_temperature,
            regularization_lambda=ccd_regular_lambda,
            alignment_lambda=ccd_alignment_lambda,
        )

    def forward(
        self, x: torch.Tensor, channel_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """_summary_

        Args:
            x (torch.Tensor): shape(batch_size * patch_num, num_features, dim)
            mask (Optional[torch.Tensor], optional): 通道掩码矩阵

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: _description_
        """
        h = self.num_heads
        # (batch_size * patch_num, num_features, dim)
        #                         ↓
        # (batch_size * patch_num, num_features, d_head * num_heads)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        scale = math.sqrt(self.d_head)

        # b: batch size
        # n: num_features(NOTE: 感觉这里是将特征维数视为了 seq_len)
        # h: num_heads
        # d: d_head
        q = einops.rearrange(q, "b n (h d) -> b h n d", h=h)
        k = einops.rearrange(k, "b n (h d) -> b h n d", h=h)
        v = einops.rearrange(v, "b n (h d) -> b h n d", h=h)

        # 计算相似度 Q * K^T
        # shape: (b, h, n, d) * (b, h, d, n) -> (b, h, n, n)
        sims = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        ccd_loss = torch.tensor(0.0, device=x.device)

        if channel_mask is not None:  # channel_mask 是 GATChannelMasker 输出的软掩码 (b, n, n)
            # 1. 使用软掩码调整CFM的注意力
            # NOTE: 加法, 软掩码对数
            masked_sims = (sims / scale) + torch.log(channel_mask.unsqueeze(1) + 1e-9)
            # 2. 计算CCD损失
            ccd_loss = self.ccd_loss_fn(sims, channel_mask)
        else:
            masked_sims = sims / scale

        # 计算注意力权重
        attention_percents = F.softmax(masked_sims, dim=-1)

        # 计算注意力分数
        attention_scores = torch.einsum("b h i j, b h j d -> b h i d", attention_percents, v)
        attention_scores = einops.rearrange(attention_scores, "b h n d -> b n (h d)")

        # (b, n, h * d) -> (b, n, dim)
        output_scores = self.to_out(attention_scores)

        return output_scores, ccd_loss


class ChannelMaskedTransformerBlocks(nn.Module):
    """注册所有的 Blocks"""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        d_head: int,
        d_ff: int,
        dropout: float = 0.8,
        ccd_temperature: float = 0.1,
        ccd_regular_lambda: float = 0.1,
        ccd_alignment_lambda: float = 1.0,
    ):
        """_summary_

        Args:
            dim (int): 模型的特征维度
            depth (int): Transformer 层数(每一层 = 预归一化的注意力层 + 预归一化的前馈网络)
            num_heads (int): 多头注意力机制的头数
            d_head (int): 每个头的特征维度
            d_ff (int): 前馈网络的隐藏层维度
            dropout (float, optional): 丢弃率 (默认 0.8)
            regular_lambda (float, optional): 动态对比损失的正则化系数 (默认 0.3)
            temperature (float, optional): 动态对比损失的温度参数 (默认 0.1)
        """
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            normalized_shape=dim,
                            module=ChannelMaskedAttention(
                                dim=dim,
                                num_heads=num_heads,
                                d_head=d_head,
                                dropout=dropout,
                                ccd_temperature=ccd_temperature,
                                ccd_regular_lambda=ccd_regular_lambda,
                                ccd_alignment_lambda=ccd_alignment_lambda,
                            ),
                        ),
                        PreNorm(
                            normalized_shape=dim,
                            module=FeedForward(
                                dim=dim,
                                hidden_dim=d_ff,
                                dropout=dropout,
                            ),
                        ),
                    ]
                )
            )

    def forward(
        self, x: torch.Tensor, channel_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """_summary_

        Args:
            x (torch.Tensor): shape: (batch_size * patch_num, num_features, dim) NOTE: patch_size * 2 映射 dim
            mask (Optional[torch.Tensor], optional): shape: (batch_size * patch_num, num_features, num_features)

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: _description_
        """
        # TODO: 这里有个问题, mask 一定传入, 不然 loss 就没了
        total_loss = torch.tensor(0.0, device=x.device)  # 动态对比损失

        for layer in self.layers:
            attention_layer, feed_forward_layer = layer  # type: ignore

            attention_scores, ccd_loss = attention_layer(
                x,
                channel_mask=channel_mask,  # NOTE: **kwargs: mask
            )

            total_loss = total_loss + ccd_loss

            # Replace in-place operations with out-of-place operations
            x = x + attention_scores  # Residual connection 1
            x = x + feed_forward_layer(x)  # Feedforward + Residual connection 2

        ccd_loss = total_loss / len(self.layers)  # 每一层平均动态对比损失
        # NOTE: x shape: (batch_size * patch_num, num_features, dim)
        return x, ccd_loss


class ChannelMaskedTransformer(nn.Module):
    """_summary_"""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        d_head: int,
        dropout: float,
        patch_dim: int,
        d_model: int,
        ccd_temperature: float = 0.1,
        ccd_regular_lambda: float = 0.1,
        ccd_alignment_lambda: float = 1.0,
    ):
        """_summary_

        Args:
            dim (int): TODO: 内部特征维度 (configs.cf_dim), Transformer 编码器处理特征的维度(嵌入维度?)
            depth (int): Transformer 层数 (configs.e_layers)
            num_heads (int): 多头注意力机制的头数 (configs.n_heads)
            d_ff (int): 前馈网络的隐藏层维度 (configs.d_ff)
            d_head (int): 每个头的特征维度 (configs.d_head)
            dropout (float): 丢弃率, 防止过拟合 (默认 0.8)
            patch_dim (int): patch_size * 2
            d_model (int): TODO: 输出维度 configs.d_model * 2 (通常是模型维度的两倍, 因为处理复数表示)
            regular_lambda (float, optional): _description_. Defaults to 0.3.
            temperature (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()

        # NOTE: patch_size * 2 嵌入为 dim
        self.patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout_layer = nn.Dropout(p=dropout)

        self.channel_transformer_blocks = ChannelMaskedTransformerBlocks(
            dim=dim,
            depth=num_layers,
            num_heads=num_heads,
            d_head=d_head,
            d_ff=d_ff,
            dropout=dropout,
            ccd_temperature=ccd_temperature,
            ccd_regular_lambda=ccd_regular_lambda,
            ccd_alignment_lambda=ccd_alignment_lambda,
        )

        self.mlp_head = nn.Linear(dim, d_model)  # horizon

    def forward(
        self, x: torch.Tensor, channel_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """_summary_

        Args:
            x (torch.Tensor): shape: (batch_size * patch_num, extended_num_features, patch_size)
            mask (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: _description_
        """
        # 将原始频域特征投影<嵌入>到 Transformer 的内部特征维度
        # patch_size -> dim
        # shape: (batch_size * patch_num, extended_num_features, dim)
        x = self.patch_embedding(x)

        # Transformer
        # -> (batch_size * patch_num, extended_num_features, dim)
        x, ccd_loss = self.channel_transformer_blocks(x, channel_mask)

        x = self.dropout_layer(x)

        # -> (batch_size * patch_num, extended_num_features, d_model)
        x = self.mlp_head(x)

        return x, ccd_loss
