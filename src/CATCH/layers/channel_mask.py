import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import gumbel_softmax

# TODO: 掩码机制: 支持基于输入数据动态生成的通道掩码

# TODO: 通道之间的掩码规则如何制定? 随机采样?


class ChannelMaskGenerator(torch.nn.Module):
    def __init__(self, patch_size: int, num_features: int):
        """_summary_

        Args:
            patch_size (int): patch 大小(对应实部/虚部), 因此后面的 patch_size * 2
            num_features (int): 特征数量(通道数)
        """
        super().__init__()

        # 掩码生成器, 用于生成通道连接概率矩阵
        self.mask_generator = nn.Sequential(
            torch.nn.Linear(patch_size * 2, num_features, bias=False),  # (in, out)
            nn.Sigmoid(),  # NOTE: Sigmoid 激活函数 -> [0, 1] 概率值
        )

        # NOTE: 手动初始化? 不记录梯度
        with torch.no_grad():
            # NOTE: sigmod(wx + b) -> sigmod(0) = 0.5 表示通道相关概率中等
            self.mask_generator[0].weight.zero_()

        self.num_features = num_features

    def _bernoulli_gumbel_rsample(self, distribution_matrix: torch.Tensor) -> torch.Tensor:
        b, c, d = distribution_matrix.shape  # (batch_size * patch_num, num_features, num_features)

        # (batch_size * patch_num * num_features * num_features, 1)
        flatten_matrix = rearrange(distribution_matrix, "b c d -> (b c d) 1")  # 展平概率 p
        r_flatten_matrix = 1 - flatten_matrix  # 反概率 (1-p)

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)  # log(p/(1-p))
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)  # log((1-p)/p)

        # Gumbel Softmax (TODO: 确保梯度流动)
        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        # NOTE: 训练时不固定随即种子
        resample_matrix = gumbel_softmax(
            new_matrix,
            hard=True,  # TODO: one-hot 编码, 只有 0/1 两种状态
        )

        # 维度重排 -> (batch_size * patch_num, num_features, num_features)
        resample_matrix = rearrange(resample_matrix[..., 0], "(b c d) -> b c d", b=b, c=c, d=d)

        return resample_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): shape (batch_size * patch_num, num_features, patch_size * 2)

        Returns:
            torch.Tensor: _description_
        """
        # 通道连接概率矩阵: (batch_size * patch_num, num_features, num_features)
        distribution_matrix = self.mask_generator(x)

        # (batch_size * patch_num, num_features, num_features)
        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)

        inverse_eye = 1 - torch.eye(self.num_features, device=x.device)
        diag = torch.eye(self.num_features, device=x.device)

        # 爱因斯坦求和约定: 计算矩阵乘法(没有重复维度, 因此不会求和, 而是逐点相乘)
        # NOTE: 强制将对角线元素设置为 1, 表示通道自连接
        resample_matrix = torch.einsum("b c d, c d -> b c d", resample_matrix, inverse_eye) + diag

        return resample_matrix


if __name__ == "__main__":
    torch.manual_seed(1037)
    x = torch.randn(2, 3, 4)  # (batch_size, num_features, patch_size * 2)
    channel_mask_generator = ChannelMaskGenerator(patch_size=2, num_features=3)
    resample_matrix = channel_mask_generator(x)
    print(resample_matrix)
