from typing import List, Tuple

import ptwt
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        assert level >= 2, "For 1-level please use WaveletPatcher."
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
            x = F.pad(x, (0, pad_len))  # F.pad 在最后一维右侧补 0： (left, right)
        return x, pad_len

    def _upsample_coeffs(self, coeffs: List[torch.Tensor], target_len: int) -> torch.Tensor:
        """
        把不同尺度的系数上采样到原始长度

        Args:
            coeffs (List[torch.Tensor]): 不同尺度的系数
            target_len (int): 原始长度

        Returns:
            torch.Tensor: 上采样后的系数
        """
        upsampled = []
        # cA_L, cD_L, cD_{L-1}, ..., cD_1
        for k, c in enumerate(coeffs):
            if k == 0:  # cA_L 近似系数
                factor = 2**self.level
            else:  # cD_{level + 1 - k} 细节系数
                factor = 2 ** (self.level - k + 1)

            # (B, C, T_small) -> repeat -> (B, C, T_small * factor)
            c_up = c.repeat_interleave(repeats=factor, dim=-1)
            c_up = c_up[..., :target_len]  # 裁到 target_len
            upsampled.append(c_up)
        # (B, C, level + 1, T_target)
        return torch.stack(upsampled, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (batch_size, seq_len, num_features)

        Returns:
            torch.Tensor: _description_
        """

        # 维度重排
        x = x.permute(0, 2, 1)  #  (B, C, T)

        # 边界填充
        x, pad_len = self._pad_to_pow2(x)  #  (B, C, T_padded)
        T_padded = x.size(-1)

        # 多层 DWT: coeffs = [cA_L, cD_L, ..., cD_1]
        coeffs = ptwt.wavedec(
            x, wavelet=self.wavelet, level=self.level, mode=self.mode  # type:ignore
        )

        # 上采样: -> (B, C, level + 1, T_padded)
        stacked = self._upsample_coeffs(coeffs, T_padded)

        # 裁掉 padding -> 恢复原序列长度
        if pad_len > 0:
            stacked = stacked[..., :-pad_len]

        # 频域 patching
        # (B, C, level + 1, T_padded) -> (B, C, level + 1, patch_num, patch_size)
        stacked = stacked.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)

        # 维度重排 -> (B, patch_num, C, level + 1, patch_size)
        stacked = stacked.permute(0, 3, 1, 2, 4)

        batch_size, patch_num, num_features, level, patch_size = stacked.shape
        # -> (batch_size * patch_num, num_features, (level + 1) * patch_size)
        z_out = stacked.reshape(batch_size * patch_num, num_features, level * patch_size)
        return z_out
