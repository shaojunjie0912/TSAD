from typing import Literal

import torch
import torch.nn as nn


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
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        elif mode == "transform":
            x = self._normalize(x)
        else:
            raise NotImplementedError
        return x

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
