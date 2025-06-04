import torch
import torch.nn as nn


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
