from typing import List, Tuple

import ptwt
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLevelWaveletPatcher(nn.Module):
    r"""Multi–level DWT patcher compatible with the CATCH forward pipeline.

    Args:
        patch_size      : 每个 patch 取多少“频点”
        patch_stride    : 滑窗步长
        level           : 小波分解层数 L (≥2)
        wavelet         : 小波基名 ('db4' / 'sym5' / 'haar' ...)
        mode            : 边界填充策略，通常 'symmetric'
    Output:
        z_out: Tensor of shape
               (batch_size * patch_num, num_features * (level+1), patch_size)
    """

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
        pow2 = 1 << self.level  # 2^level
        pad_len = (pow2 - seq_len % pow2) % pow2
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))  # F.pad 在最后一维右侧补 0： (left, right)
        return x, pad_len

    # ------------------------------------------------------------------ #
    # Helper:   把不同尺度的系数 ↑ 上采到原始长度  (repeat_interleave)
    #           cA_L scale factor = 2**level ;  cD_l = 2**(l-1)
    # ------------------------------------------------------------------ #
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
        # coeffs[0] = cA_L,  coeffs[1] = cD_L, ..., coeffs[-1] = cD_1
        for k, c in enumerate(coeffs):
            # k == 0  → cA_L
            # k >= 1  → cD_{level+1-k}
            factor = 1 << (self.level if k == 0 else self.level - k)
            # (B, C, T_small) → repeat → (B, C, T_small*factor)
            c_up = c.repeat_interleave(factor, dim=-1)
            # 裁到 target_len（repeat 可能会多出 1-factor 处）
            c_up = c_up[..., :target_len]
            upsampled.append(c_up)
        # 在“通道维” stack → (B, C*(level+1), T)
        return torch.cat(upsampled, dim=1)

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

        # 3) 多层 DWT
        #    coeffs = [cA_L, cD_L, ..., cD_1]
        coeffs = ptwt.wavedec(
            x, wavelet=self.wavelet, level=self.level, mode=self.mode  # type:ignore
        )

        # 4) ↑ Upsample 到原 T_padded, 然后在通道维 concat
        #    stacked: (B, C*(level+1), T_padded)
        stacked = self._upsample_coeffs(coeffs, T_padded)

        # 5) 裁掉 padding → 恢复原序列长度  (B, C*(level+1), T)
        if pad_len > 0:
            stacked = stacked[..., :-pad_len]

        # 6) 频域 patching  —— 在 -1 维滑窗
        #    (B, C', T) → (B, C', patch_num, patch_size)
        stacked = stacked.unfold(
            dimension=-1, size=self.patch_size, step=self.patch_stride
        )  #  (B, C', patch_num, patch_size)

        # 7) 维度重排  (B, patch_num, C', patch_size)
        stacked = stacked.permute(
            0, 2, 1, 3
        )  #  (batch_size, patch_num, num_features*(L+1), patch_size)

        # 8) 合并 batch & patch_num → (B*patch_num, C', patch_size)
        B, patch_num, C_channels, _ = stacked.shape
        z_out = stacked.reshape(B * patch_num, C_channels, self.patch_size)

        #  (batch_size * patch_num, num_features*(level+1), patch_size)
        return z_out


if __name__ == "__main__":
    # (batch_size, num_features, seq_len)
    # 写一个 (1, 3, 5) 的 tensor
    x = torch.tensor(
        [
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        ],
        dtype=torch.float32,
    )
