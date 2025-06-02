from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..layers.channel_mask import ChannelMaskGenerator
from ..layers.channel_masked_transformer import ChannelMaskedTransformer
from ..layers.flatten_head import FlattenHead

#
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
        seq_len: int,
        dim: int,
        num_heads: int,
        d_head: int,
        d_ff: int,
        dropout: float,
        head_dropout: float,
        d_model: int,
        regular_lambda: float,
        temperature: float,
        flatten_individual: bool,
    ):
        """_summary_

        Args:
            num_features (int): _description_
            num_layers (int): _description_
            affine (bool): _description_
            subtract_last (bool): _description_
            patch_size (int): _description_
            patch_stride (int): _description_
            seq_len (int): _description_
            dim (int): _description_
            num_heads (int): _description_
            d_head (int): _description_
            d_ff (int): _description_
            dropout (float): _description_
            head_dropout (float): _description_
            d_model (int): _description_
            regular_lambda (float): _description_
            temperature (float): _description_
            flatten_individual (bool): _description_
        """
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
        patch_num = int((seq_len - patch_size) / patch_stride + 1)
        self.norm = nn.LayerNorm(self.patch_size)

        # Backbone
        self.re_attn = True
        self.mask_generator = ChannelMaskGenerator(patch_size=patch_size, num_features=num_features)
        self.frequency_transformer = ChannelMaskedTransformer(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            d_head=d_head,
            dropout=dropout,
            patch_dim=patch_size * 2,
            # horizon=self.horizon * 2,
            d_model=d_model * 2,  # TODO: d_model * 2
            regular_lambda=regular_lambda,
            temperature=temperature,
        )

        # Head
        self.flatten_head_real = FlattenHead(
            individual=flatten_individual,
            num_features=num_features,
            input_dim=d_model * 2 * patch_num,
            seq_len=seq_len,
            head_dropout=head_dropout,
        )
        self.flatten_head_imag = FlattenHead(
            individual=flatten_individual,
            num_features=num_features,
            input_dim=d_model * 2 * patch_num,
            seq_len=seq_len,
            head_dropout=head_dropout,
        )

        # 反傅里叶变换后的实部虚部映射
        self.ri_projection = nn.Linear(seq_len * 2, seq_len)

        # 获取实部和虚部的线性层
        # TODO: 为什么还要加一个线性层?
        self.get_real = nn.Linear(d_model * 2, d_model * 2)
        self.get_imag = nn.Linear(d_model * 2, d_model * 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """_summary_

        Args:
            x (torch.Tensor): (batch_size, seq_len, num_features) (B, T, C), 这里的 seq_len 也是滑动窗口大小

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: [时域重构结果, 频域重构结果, 动态对比损失]
        """
        # ---------------------------------------------
        # ---------------- 前向模块 (FM) -------------------
        # ---------------------------------------------
        # 实例归一化
        x = self.revin_layer(x, "norm")

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
        # unfold 滑窗, 当 (seq_len - patch_size) % patch_stride == 0 时, 能覆盖所有数据
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

        # ------------------------------------------------------
        # ---------------- 通道融合模块 (CFM) -------------------
        # ------------------------------------------------------

        # 通道掩码矩阵 (batch_size * patch_num, num_features, num_features)
        channel_mask = self.mask_generator(z_cat)

        # z_hat: (batch_size * patch_num, num_features, d_model * 2)
        z_hat, dc_loss = self.frequency_transformer(z_cat, channel_mask)

        # 提取实部和虚部特征
        z_hat_real = self.get_real(z_hat)  # (batch_size * patch_num, num_features, d_model * 2)
        z_hat_imag = self.get_imag(z_hat)  # (batch_size * patch_num, num_features, d_model * 2)

        z_hat_real = z_hat_real.reshape((batch_size, patch_num, num_features, z_hat_real.shape[-1]))
        z_hat_imag = z_hat_imag.reshape((batch_size, patch_num, num_features, z_hat_imag.shape[-1]))

        # z1: [bs, nvars， patch_num, horizon] TODO: horizon?
        # 维度重排 (batch_size, num_features, patch_num, d_model * 2)
        z_hat_real = z_hat_real.permute(0, 2, 1, 3)
        z_hat_imag = z_hat_imag.permute(0, 2, 1, 3)

        # 展平, 将分块特征重建为完整序列 (batch_size, num_features, seq_len)
        z_hat_real = self.flatten_head_real(z_hat_real)
        z_hat_imag = self.flatten_head_imag(z_hat_imag)

        # ------------ 时频重构模块 (TFRM) ----------------

        # 拼接实虚部
        z_hat = torch.complex(z_hat_real, z_hat_imag)

        # 逆傅里叶变换 TODO: 虚部都是 0j, 微小的虚部信息有用吗?
        x_hat = torch.fft.ifft(z_hat)
        x_hat_real = x_hat.real
        x_hat_imag = x_hat.imag
        # 使用线性层组合时虚部, 获得最终重建序列 2 * seq_len -> seq_len
        # (batch_size, num_features, seq_len)
        x_hat = self.ri_projection(torch.cat((x_hat_real, x_hat_imag), dim=-1))

        # denorm
        x_hat = x_hat.permute(0, 2, 1)  # 维度重排 (batch_size, seq_len, num_features)
        x_hat = self.revin_layer(x_hat, "denorm")  # 逆实例归一化

        return x_hat, z_hat.permute(0, 2, 1), dc_loss
