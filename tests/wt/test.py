from typing import List, Literal, Tuple

import ptwt
import torch
import torch.nn as nn


class WaveletPatcher(nn.Module):
    def __init__(
        self,
        wave: str,
        J: int,
        mode: Literal["constant", "zero", "reflect", "periodic", "symmetric"],
        patch_size: int,
        patch_stride: int,
    ):
        super().__init__()
        self.wave = wave  # 小波基名称
        self.J = J  # 分解层数
        self.mode = mode  # 分解模式
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    @torch.no_grad()  # patch 化本身不需梯度
    def _mra(self, x: torch.Tensor) -> List[torch.Tensor]:
        """多层分解，返回 list: [cA_J, cD_J, ..., cD_1]"""
        coeffs = ptwt.wavedec(x, wavelet=self.wave, level=self.J, mode=self.mode)  # type: ignore
        return coeffs

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)  # 维度重排 (batch_size, num_features, seq_len)
        coeffs = self._mra(x)  #


if __name__ == "__main__":
    # (batch_size, num_features, seq_len)
    # 写一个 (1, 3, 5) 的 tensor
    x = torch.tensor(
        [
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        ],
        dtype=torch.float32,
    )
