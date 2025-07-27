"""
SWIFT-AD (Stationary Wavelet-patched Inter-channel Fusion Transformer for Anomaly Detection)
Multiresolution Wavelet Patching and Graph-Guided Channel Fusion for Robust Multivariate Time-Series Anomaly Detection
基于多尺度小波变换和图引导通道融合的鲁棒多变量时间序列异常检测
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.channel_masked_transformer import ChannelMaskedTransformer
from ..layers.channel_masker import GATChannelMasker
from ..layers.channel_patcher import InverseWaveletPatcher, WaveletPatcher


class ReconstructionHead(nn.Module):
    def __init__(
        self,
        individual: bool,
        num_features: int,
        input_dim: int,
        seq_len: int,
        head_dropout: float = 0,
    ):
        """

        Args:
            individual (bool): 每个变量使用独立的网络 / 所有变量共享相同的网络
            num_features (int): 变量数量
            input_dim (int): 输入特征维度 (d_model * patch_num)
            seq_len (int): 输出序列长度
            head_dropout (float, optional): 丢弃率 (默认=0)
        """
        super().__init__()

        self.individual = individual
        self.num_features = num_features

        if self.individual:
            # 每个变量使用独立的网络, (flatten + linear + dropout)s
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.num_features):
                # NOTE: flatten 从倒数第 2 维 patch_num, d_model 开始展平
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(input_dim, seq_len))
                self.dropouts.append(nn.Dropout(p=head_dropout))
        else:
            # 所有变量共享相同的网络, flatten + (linear)s + dropout
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(input_dim, input_dim)
            self.linear2 = nn.Linear(input_dim, input_dim)
            self.linear3 = nn.Linear(input_dim, input_dim)
            self.linear4 = nn.Linear(input_dim, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (batch_size, num_features, patch_num, d_model)

        Returns:
            torch.Tensor: (batch_size, num_features, seq_len)
        """
        if self.individual:
            x_out = []
            for i in range(self.num_features):
                z = self.flattens[i](x[:, i, :, :])  # z: (batch_size, d_model * patch_num)
                z = self.linears[i](z)  # z: (batch_size, seq_len)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: (batch_size, num_features, seq_len)
        else:
            # 展平
            x = self.flatten(x)
            # 3 层残差块
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            # 投影
            x = self.linear4(x)
            # TODO: 要加入 dropout 吗?

        return x


class RevIN(nn.Module):
    """可逆实例归一化层"""

    # NOTE:
    # RevIN 通过实例归一化消除时间序列数据的分布差异，
    # 然后在预测完成后通过反向操作恢复原始分布特征，
    # 有效解决分布偏移问题，提高预测精度。
    # 这种方法特别适用于具有季节性变化或趋势的时间序列数据。

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        """

        Args:
            num_features (int): 特征数量
            eps (float, optional): 数值稳定性增强值. Defaults to 1e-5.
            affine (bool, optional): 是否使用可学习的仿射参数. Defaults to True.
            subtract_last (bool, optional): 是否用最后一个值代替均值. Defaults to False.
        """
        super().__init__()
        self.num_features = num_features  # 特征数量
        self.eps = eps  # 数值稳定性增强值
        self.affine = affine  # 是否使用可学习的仿射参数
        self.subtract_last = subtract_last  # 是否用最后一个值代替均值
        if self.affine:
            self._init_params()

    def forward(
        self, x: torch.Tensor, mode: Literal["norm", "denorm", "transform"]
    ) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): _description_
            mode (Literal[&quot;norm&quot;, &quot;denorm&quot;, &quot;transform&quot;]):
                norm: 归一化(用于模型输入前的数据归一化, 存储数据的均值和标准差)
                denorm: 反归一化
                transform: 转换(归一化)

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.Tensor: _description_
        """
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        elif mode == "transform":
            return self._normalize(x)

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor) -> None:
        """计算并存储数据的均值和标准差

        Args:
            x (torch.Tensor):
        """
        # NOTE: 对于 (B, T, C) 的输入, dim2reduce = 1
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            # NOTE:
            # x[:, -1, :]: (B, C)
            # unsqueeze(1): (B, 1, C) 在第二维(1)增加一个维度
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            # NOTE:
            # 在每个 batch 的 T 维度上计算<均值>和<标准差>(带 eps)
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        # NOTE: eps: 数值稳定性增强值
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """使用已有的均值和标准差对数据进行归一化

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """将归一化后的数据恢复到原始数据分布

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        if self.affine:
            x = x - self.affine_bias
            # TODO: eps^2? 避免除零错误, 增强数值稳定性
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# NOTE: 原来的 configs 是一个类, 所有参数都是类属性(从字典中读取并通过 setaddr 设置)
class SWIFT(nn.Module):
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
        attention_dropout: float,
        num_gat_heads: int,
        gat_head_dim: int,
        gat_dropout_rate: float,
        rec_head_dropout: float,
        d_model: int,
        ccd_align_temperature: float,
        ccd_regular_lambda: float,
        ccd_align_lambda: float,
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
            num_gat_heads=num_gat_heads,
            gat_head_dim=gat_head_dim,
            dropout_rate=gat_dropout_rate,
        )

        self.d_model = d_model
        self.channel_masked_transformer = ChannelMaskedTransformer(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            d_head=d_head,
            dropout=dropout,
            attention_dropout=attention_dropout,
            patch_dim=patch_size,
            d_model=d_model,
            ccd_temperature=ccd_align_temperature,
            ccd_regular_lambda=ccd_regular_lambda,
            ccd_alignment_lambda=ccd_align_lambda,
        )

        self.reconstruction_head = ReconstructionHead(
            individual=is_flatten_individual,
            num_features=self.extended_num_features,
            input_dim=d_model * self.patch_num,
            seq_len=seq_len,
            head_dropout=rec_head_dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """_summary_

        Args:
            x (torch.Tensor): (B, T, C) (B, T, C), 这里的 T 也是滑动窗口大小

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: [时域重构结果, 频域重构结果, 动态对比损失]
        """
        x_original = x.clone()
        B = x.size(0)
        # ---------------------------------------------
        # ---------------- 前向模块 (FM) --------------
        # ---------------------------------------------
        # 实例归一化
        x_norm = self.revin_layer(x, "norm")

        # patching 小波变换 -> (B * patch_num, Extd_C, patch_size)
        # Extd_C = C * (L + 1)
        original_coeffs, z_out = self.patcher(x_norm)

        # ------------------------------------------------------
        # ---------------- 通道融合模块 (CFM) ------------------
        # ------------------------------------------------------

        # 通道掩码矩阵 (B * patch_num, Extd_C, Extd_C)
        channel_mask = self.masker(z_out)

        # z_hat: (B * patch_num, Extd_C, d_model)
        z_hat, ccd_loss = self.channel_masked_transformer(z_out, channel_mask)

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

        return (
            x_original,  # x: 时间域原始值 (B, T, C)
            x_hat,  # x_hat: 时间域重构值 (B, T, C)
            original_coeffs.permute(0, 2, 1),  # original_coeffs: 尺度域原始值 (B, T, Extd_C)
            reconstructed_coeffs.permute(
                0, 2, 1
            ),  # reconstructed_coeffs: 尺度域重构值 (B, T, Extd_C)
            ccd_loss,  # ccd_loss: 通道相关性发掘损失
        )
