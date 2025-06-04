from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from ..layers.channel_masked_transformer import ChannelMaskedTransformer
from ..layers.channel_masker import GATChannelMasker
from ..layers.flatten_head import FlattenHead
from ..layers.patcher import FftPatcher, WaveletPatcher
from ..layers.RevIN import RevIN

# TODO: 最外层的参数名称可以更加清晰完整


# NOTE: 原来的 configs 是一个类, 所有参数都是类属性(从字典中读取并通过 setaddr 设置)
class CATCH(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_layers: int,
        affine: bool,
        subtract_last: bool,
        patch_size: int,
        patch_stride: int,
        level: int,
        wavelet: str,
        mode: str,
        seq_len: int,
        dim: int,
        num_heads: int,
        d_head: int,
        d_ff: int,
        dropout: float,
        head_dropout: float,
        d_model: int,
        ccd_temperature: float,
        ccd_regular_lambda: float,
        ccd_alignment_lambda: float,
        is_flatten_individual: bool,
    ):
        super().__init__()
        self.num_features = num_features
        self.revin_layer = RevIN(
            num_features=num_features,
            affine=affine,
            subtract_last=subtract_last,
        )

        # Patching
        # TODO: patch_size 和 patch_stride 目前都是固定的
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_num = int((seq_len - patch_size) / patch_stride + 1)
        self.patcher: nn.Module
        new_patch_size: int
        self.patcher = WaveletPatcher(
            patch_size=patch_size,
            patch_stride=patch_stride,
            level=level,
            wavelet=wavelet,
            mode=mode,
        )
        new_patch_size = patch_size * (level + 1)
        self.norm = nn.LayerNorm(self.patch_size)
        self.mask_generator = GATChannelMasker(
            node_feature_dim=new_patch_size,
            num_features=num_features,
        )

        # Backbone
        self.re_attn = True

        self.d_model = d_model
        self.frequency_transformer = ChannelMaskedTransformer(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            d_head=d_head,
            dropout=dropout,
            patch_dim=new_patch_size,
            # horizon=self.horizon * 2,
            d_model=d_model,  # TODO: d_model * 2
            ccd_temperature=ccd_temperature,
            ccd_regular_lambda=ccd_regular_lambda,
            ccd_alignment_lambda=ccd_alignment_lambda,
        )

        # Head
        self.flatten_head = FlattenHead(
            individual=is_flatten_individual,
            num_features=num_features,
            input_dim=d_model * self.patch_num,  # TODO: d_model * 2 * self.patch_num
            seq_len=seq_len,
            head_dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """_summary_

        Args:
            x (torch.Tensor): (batch_size, seq_len, num_features) (B, T, C), 这里的 seq_len 也是滑动窗口大小

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: [时域重构结果, 频域重构结果, 动态对比损失]
        """
        batch_size = x.size(0)
        # ---------------------------------------------
        # ---------------- 前向模块 (FM) --------------
        # ---------------------------------------------
        # 实例归一化
        x = self.revin_layer(x, "norm")

        # patching (傅里叶变换 FFT /小波变换 WT)
        # z_cat:
        # FFT: (batch_size * patch_num, num_features, patch_size * 2)
        # WT: (batch_size * patch_num, num_features, (level + 1) * patch_size)
        z_cat = self.patcher(x)

        # ------------------------------------------------------
        # ---------------- 通道融合模块 (CFM) ------------------
        # ------------------------------------------------------

        # 通道掩码矩阵 (batch_size * patch_num, num_features, num_features)
        channel_mask = self.mask_generator(z_cat)

        # z_hat: (batch_size * patch_num, num_features, d_model * 2)
        z_hat, dc_loss = self.frequency_transformer(z_cat, channel_mask)

        # -------------------------------------------------
        # ------------ 时频重构模块 (TFRM) ----------------
        # -------------------------------------------------

        z_hat = z_hat.reshape(batch_size, self.patch_num, self.num_features, self.d_model)
        z_hat = z_hat.permute(0, 2, 1, 3)

        # 展平
        z_hat = self.flatten_head(z_hat)
        x_hat = z_hat.permute(0, 2, 1)  # 维度重排 (batch_size, seq_len, num_features)
        x_hat = self.revin_layer(x_hat, "denorm")  # 逆实例归一化

        return x_hat, z_hat.permute(0, 2, 1), dc_loss
