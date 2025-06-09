from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..layers.channel_masked_transformer import ChannelMaskedTransformer
from ..layers.channel_masker import GATChannelMasker
from ..layers.channel_patcher import InverseWaveletPatcher, WaveletPatcher
from ..layers.preprocessor import ReconstructionHead, RevIN


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

        self.patcher = WaveletPatcher(
            patch_size=patch_size,
            patch_stride=patch_stride,
            level=level,
            wavelet=wavelet,
            mode=mode,
        )

        self.inverse_patcher = InverseWaveletPatcher(
            level=level,
            wavelet=wavelet,
            mode=mode,
        )

        self.norm = nn.LayerNorm(patch_size)

        self.extended_num_features = num_features * (level + 1)

        self.masker = GATChannelMasker(
            node_feature_dim=patch_size,
            num_features=self.extended_num_features,
        )

        self.d_model = d_model
        self.channel_masked_transformer = ChannelMaskedTransformer(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            d_head=d_head,
            dropout=dropout,
            patch_dim=patch_size,
            # horizon=self.horizon * 2,
            d_model=d_model,  # TODO: d_model * 2
            ccd_temperature=ccd_temperature,
            ccd_regular_lambda=ccd_regular_lambda,
            ccd_alignment_lambda=ccd_alignment_lambda,
        )

        self.reconstruction_head = ReconstructionHead(
            individual=is_flatten_individual,
            num_features=self.extended_num_features,
            input_dim=d_model * self.patch_num,
            seq_len=seq_len,
            head_dropout=head_dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """_summary_

        Args:
            x (torch.Tensor): (B, T, C) (B, T, C), 这里的 T 也是滑动窗口大小

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: [时域重构结果, 频域重构结果, 动态对比损失]
        """
        B = x.size(0)
        # ---------------------------------------------
        # ---------------- 前向模块 (FM) --------------
        # ---------------------------------------------
        # 实例归一化
        x_norm = self.revin_layer(x, "norm")

        # patching 小波变换 -> (B * patch_num, Extd_C, patch_size)
        # Extd_C = C * (L + 1)
        original_coeffs, z_cat = self.patcher(x_norm)

        # ------------------------------------------------------
        # ---------------- 通道融合模块 (CFM) ------------------
        # ------------------------------------------------------

        # 通道掩码矩阵 (B * patch_num, Extd_C, Extd_C)
        channel_mask = self.masker(z_cat)

        # z_hat: (B * patch_num, Extd_C, d_model)
        z_hat, ccd_loss = self.channel_masked_transformer(z_cat, channel_mask)

        # -------------------------------------------------
        # ------------ 时尺重构模块 (TSRM) ----------------
        # -------------------------------------------------

        # -> (B, Extd_C, patch_num, d_model)
        z_hat = z_hat.reshape(B, self.patch_num, self.extended_num_features, self.d_model)
        z_hat = z_hat.permute(0, 2, 1, 3)

        # (B, Extd_C, T)
        reconstructed_coeffs = self.reconstruction_head(z_hat)

        # 逆小波变换
        # -> (B, C, T)
        x_hat_norm = self.inverse_patcher(reconstructed_coeffs)

        # 维度重排 (B, T, C)
        x_hat_norm = x_hat_norm.permute(0, 2, 1)
        x_hat = self.revin_layer(x_hat_norm, "denorm")  # 逆实例归一化

        # x: 时间域原始值 (B, T, C)
        # x_hat: 时间域重构值 (B, T, C)
        # original_coeffs: 尺度域原始值 (B, T, Extd_C)
        # reconstructed_coeffs: 尺度域重构值 (B, T, Extd_C)
        # ccd_loss: 通道相关性发掘损失
        return (
            x_hat,
            original_coeffs.permute(0, 2, 1),
            reconstructed_coeffs.permute(0, 2, 1),
            ccd_loss,
        )
