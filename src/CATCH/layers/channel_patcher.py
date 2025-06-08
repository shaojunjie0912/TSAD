from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptwt.stationary_transform import swt


class FftPatcher(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (batch_size, seq_len, num_features)

        Returns:
            torch.Tensor: _description_
        """
        # 傅里叶变换 FFT
        x = x.permute(0, 2, 1)  # 维度重排 (batch_size, num_features, seq_len) (B, C, T)
        z = torch.fft.fft(x)  # 默认维度 -1(最后一个), 即对 seq_len 个时间步进行傅里叶变换
        z_real = z.real  # 实部
        z_imag = z.imag  # 虚部

        # 频域切片(patching) -> 频带(frequency patches)
        # 分别在实部和虚部的最后一个维度(频率)上进行切片(滑窗)
        # (batch_size, num_features, seq_len)
        #                  ↓
        # (batch_size, num_features, patch_num, patch_size)
        # unfold: 滑动窗口提取
        z_real = z_real.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        z_imag = z_imag.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)

        # 维度重排 (batch_size, patch_num, num_features, patch_size)
        z_real = z_real.permute(0, 2, 1, 3)
        z_imag = z_imag.permute(0, 2, 1, 3)

        batch_size = z_real.shape[0]
        patch_num = z_real.shape[1]
        num_features = z_real.shape[2]
        patch_size = z_real.shape[3]

        # 形状变换 (batch_size * patch_num, num_features, patch_size)
        z_real = z_real.reshape(batch_size * patch_num, num_features, patch_size)
        z_imag = z_imag.reshape(batch_size * patch_num, num_features, patch_size)

        # 拼接实部和虚部 [实部, 虚部] NOTE: 并非实虚交错 TODO: 真的存在数学原理吗?
        # z_cat: (batch_size * patch_num, num_features, patch_size * 2)
        z_cat = torch.cat((z_real, z_imag), dim=-1)

        return z_cat


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

    def _pad_to_pow2(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        将序列长度 padding 到 2^level 的倍数

        Args:
            x (torch.Tensor): 输入序列

        Returns:
            Tuple[torch.Tensor, int]: 填充后的序列和填充长度
        """
        seq_len = x.size(-1)
        pow2 = 2**self.level  # 2^level
        pad_len = (pow2 - seq_len % pow2) % pow2
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode="replicate")
        return x, pad_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (batch_size, seq_len, num_features)

        Returns:
            torch.Tensor: (batch_size * patch_num,(level + 1) * num_features,  patch_size)
        """

        # 维度重排
        B, T, N = x.shape
        x = x.permute(0, 2, 1)  #  (B, C, T)

        # 边界填充
        x, pad_len = self._pad_to_pow2(x)  #  (B, C, T_padded)
        T_pad = x.size(-1)

        # SWT
        xn = x.reshape(B * N, T_pad)  # (B·N, T_pad)
        # list[(B·N, T_pad)], len=L+1
        coeffs = swt(xn, wavelet=self.wavelet, level=self.level, axis=-1)

        # 顺序: [A_L, D_L, D_{L-1}, …, D_1]
        coeff_stack = torch.stack(coeffs, dim=1)  # (B·N, L+1, T_pad)
        coeff_stack = coeff_stack.view(B, N, self.level + 1, T_pad)  # (B, N, L+1, T_pad)

        # -------- remove padding --------
        if pad_len:
            coeff_stack = coeff_stack[..., :-pad_len]

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
        z_out = coeff_stack.view(B * P, N * Lp1, p)  # (B·P, (L+1)·N, p)
        return z_out
