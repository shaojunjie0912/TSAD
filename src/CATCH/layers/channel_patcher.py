from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptwt.stationary_transform import iswt, swt


class WaveletPatcher(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_stride: int,
        level: int = 3,
        wavelet: str = "db4",
        mode: str = "symmetric",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.level = level
        self.wavelet = wavelet
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (batch_size, seq_len, num_features)

        Returns:
            torch.Tensor: (batch_size * patch_num,(level + 1) * num_features,  patch_size)
        """

        # 维度重排
        B, T, N = x.shape
        if T % (2**self.level) != 0:
            raise ValueError(f"seq_len {T} must be divisible by 2^{self.level}")
        x = x.permute(0, 2, 1)  #  (B, C, T)

        # SWT
        xn = x.reshape(B * N, T)  # (B·N, T)
        # list[(B·N, T)], len=L+1
        coeffs = swt(xn, wavelet=self.wavelet, level=self.level, axis=-1)

        # 顺序: [A_L, D_L, D_{L-1}, …, D_1]
        coeff_stack = torch.stack(coeffs, dim=1)  # (B·N, L+1, T)
        coeff_stack = coeff_stack.reshape(B, N, self.level + 1, T)  # (B, N, L+1, T)

        # -------- patch along time --------
        # (B, N, L+1, T) ➜ unfold ➜ (B, N, L+1, P, p)
        coeff_stack = coeff_stack.unfold(
            dimension=-1,
            size=self.patch_size,
            step=self.patch_stride,
        )

        # -------- reshape: (B, P, N, L+1, p) --------
        coeff_stack = coeff_stack.permute(0, 3, 1, 2, 4)
        B, P, N, Lp1, p = coeff_stack.shape

        # -------- final output --------
        z_out = coeff_stack.reshape(B * P, N * Lp1, p)  # (B·P, (L+1)·N, p)
        return z_out


class InverseWaveletPatcher(nn.Module):
    def __init__(self, level: int, wavelet: str, mode: str):
        """
        通过逆平稳小波变换 (iSWT) 将小波系数重构为时域信号。

        Args:
            level (int): 小波分解的层数。
            wavelet (str): 使用的小波基名称, e.g., 'db4'.
            mode (str): 信号延拓模式。
        """
        super().__init__()
        self.level = level
        self.wavelet = wavelet
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行逆小波变换。

        Args:
            x (torch.Tensor): 输入的重构后的小波系数张量。
                              形状: (batch_size, (level + 1) * num_features, seq_len)

        Returns:
            torch.Tensor: 重构后的时域信号。
                          形状: (batch_size, num_features, seq_len)
        """
        B, expanded_N, T = x.shape
        N = expanded_N // (self.level + 1)  # 计算原始通道数

        # 将扩展通道维度分解为 (原始通道数, 小波系数层数)
        # (B, (L+1)*N, T) -> (B, N, L+1, T)
        x = x.reshape(B, N, self.level + 1, T)

        # 准备 iswt 的输入格式
        # (B, N, L+1, T) -> (B*N, L+1, T)
        x = x.reshape(B * N, self.level + 1, T)

        # iswt 需要一个系数列表 [A_L, D_L, ..., D_1]
        # 将张量在第1个维度 (L+1) 上分割成列表
        coeffs_list = [c.squeeze(1) for c in torch.split(x, 1, dim=1)]

        # 执行逆平稳小波变换
        # (B*N, T)
        reconstructed_signal = iswt(coeffs_list, wavelet=self.wavelet)

        # 恢复原始的 batch 和通道维度
        # (B*N, T) -> (B, N, T)
        reconstructed_signal = reconstructed_signal.reshape(B, N, T)

        return reconstructed_signal
