import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax

# TODO: 掩码机制: 支持基于输入数据动态生成的通道掩码

# TODO: 通道之间的掩码规则如何制定? 随机采样?


class Projector(torch.nn.Module):
    def __init__(self, new_patch_size: int, num_features: int):
        super().__init__()
        self.linear_layer = torch.nn.Linear(new_patch_size, num_features, bias=False)  # (in, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class LinearChannelMasker(torch.nn.Module):
    def __init__(self, new_patch_size: int, num_features: int):
        """_summary_

        Args:
            new_patch_size (int): patch 大小, FFT 和 WT 不同
            num_features (int): 特征数量(通道数)
        """
        super().__init__()

        # 掩码生成器, 用于生成通道连接概率矩阵
        self.projector = Projector(new_patch_size, num_features)

        # NOTE: 手动初始化? 不记录梯度
        with torch.no_grad():
            # NOTE: sigmod(wx + b) -> sigmod(0) = 0.5 表示通道相关概率中等
            self.projector.linear_layer.weight.zero_()

        self.mask_generator = nn.Sigmoid()  # NOTE: Sigmoid 激活函数 -> [0, 1] 概率值
        self.num_features = num_features

    def _bernoulli_gumbel_rsample(self, distribution_matrix: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            distribution_matrix (torch.Tensor): (batch_size * patch_num, num_features, num_features)

        Returns:
            torch.Tensor: _description_
        """
        b, c, d = distribution_matrix.shape

        # (batch_size * patch_num * num_features * num_features, 1)
        flatten_matrix = einops.rearrange(distribution_matrix, "b c d -> (b c d) 1")  # 展平概率 p
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
        resample_matrix = einops.rearrange(
            resample_matrix[..., 0], "(b c d) -> b c d", b=b, c=c, d=d
        )

        return resample_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): shape (batch_size * patch_num, num_features, new_patch_size)

        Returns:
            torch.Tensor: _description_
        """
        projected_x = self.projector(x)
        # 通道连接概率矩阵: (batch_size * patch_num, num_features, num_features)
        distribution_matrix = self.mask_generator(projected_x)

        # (batch_size * patch_num, num_features, num_features)
        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)

        inverse_eye = 1 - torch.eye(self.num_features, device=x.device)
        diag = torch.eye(self.num_features, device=x.device)

        # 爱因斯坦求和约定: 计算矩阵乘法(没有重复维度, 因此不会求和, 而是逐点相乘)
        # NOTE: 强制将对角线元素设置为 1, 表示通道自连接
        resample_matrix = torch.einsum("b c d, c d -> b c d", resample_matrix, inverse_eye) + diag

        return resample_matrix


class GATChannelMasker(nn.Module):
    """
    使用图注意力网络 (GAT) 的思想为通道生成软掩码。
    每个 patch 内的 C 个通道被视为图的节点。
    节点特征是该通道在当前 patch 内的特征表示。
    GAT 用于学习通道间的注意力权重, 这些权重作为软掩码
    """

    def __init__(
        self,
        node_feature_dim: int,
        num_features: int,
        num_gat_heads: int = 4,
        gat_head_dim: int = 16,
        dropout_rate: float = 0.1,
    ):
        """
        初始化 GATChannelMasker。

        Args:
            node_feature_dim (int): 每个通道节点输入的特征维度 (来自 WaveletPatcher, 即 new_patch_size)
            num_channels (int): 通道数量 (即图中的节点数量)
            num_gat_heads (int): GAT 中的注意力头数
            gat_head_dim (int): 每个GAT注意力头的维度
            dropout_rate (float): Attention权重的Dropout比率
        """
        super().__init__()
        self.num_channels = num_features
        self.num_gat_heads = num_gat_heads
        self.gat_head_dim = gat_head_dim

        self.inner_dim = self.gat_head_dim * self.num_gat_heads  # Q, K 的总维度

        # 线性变换层，用于计算 Query 和 Key
        self.to_q = nn.Linear(node_feature_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(node_feature_dim, self.inner_dim, bias=False)

        self.attention_dropout = nn.Dropout(dropout_rate)

        # 初始化权重，使得初始的注意力分布是均匀的
        # 即 Q 和 K 初始为0, 使得相似度为0, softmax后为均匀分布
        with torch.no_grad():
            self.to_q.weight.zero_()
            self.to_k.weight.zero_()

    def forward(self, x_nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_nodes (torch.Tensor): 节点特征张量。
                                   形状: (B_eff, num_channels, node_feature_dim)
                                   B_eff = batch_size * patch_num

        Returns:
            torch.Tensor: 计算得到的软掩码 (通道间注意力权重)。
                          形状: (B_eff, num_channels, num_channels)
                          值范围 [0, 1], 代表通道i对通道j的注意力。
        """
        # x_nodes: (B_eff, N, D_node_feat) where N = num_channels

        # 线性投影得到 Q 和 K
        # q_channels, k_channels: (B_eff, N, inner_dim)
        q_channels = self.to_q(x_nodes)
        k_channels = self.to_k(x_nodes)

        # 重塑以支持多头注意力
        # q_channels, k_channels: (B_eff, H_gat, N, D_head_gat)
        q_channels = einops.rearrange(
            q_channels, "b n (h d) -> b h n d", h=self.num_gat_heads, d=self.gat_head_dim
        )
        k_channels = einops.rearrange(
            k_channels, "b n (h d) -> b h n d", h=self.num_gat_heads, d=self.gat_head_dim
        )

        # 计算缩放点积注意力原始分数 (相似度)
        # scale 因子
        scale = math.sqrt(self.gat_head_dim)
        # channel_sim: (B_eff, H_gat, N, N) (通道i的Q 与 通道j的K 的相似度)
        channel_sim = torch.einsum("b h i d, b h j d -> b h i j", q_channels, k_channels) / scale

        # 应用 Softmax 得到注意力权重 (P(j|i))
        # channel_att_weights: (B_eff, H_gat, N, N)
        channel_att_weights = F.softmax(channel_sim, dim=-1)  # 在最后一个维度（key维度）上softmax
        channel_att_weights = self.attention_dropout(channel_att_weights)

        # 平均多个头的注意力权重得到最终的软掩码
        # soft_mask: (B_eff, N, N)
        soft_mask = channel_att_weights.mean(dim=1)

        # （可选）确保对角线为1或较高值，表示强自相关。
        # 一种简单方式是直接赋值，但这会破坏softmax的sum-to-1特性（如果后续需要）。
        # 另一种方式是在计算channel_sim时给对角线加上一个偏置。
        # 为了简单起见并允许模型学习自相关强度，我们暂时不强制修改对角线。
        # 如果需要强制自环，可以在CATCH的CFM的ChannelMaskedAttention中，在使用mask时特殊处理对角线。
        # 或者在这里后处理，例如:
        # soft_mask_diag = torch.eye(self.num_channels, device=soft_mask.device, dtype=soft_mask.dtype).unsqueeze(0)
        # soft_mask = soft_mask * (1 - soft_mask_diag) + soft_mask_diag # 强制对角线为1，非对角线为学习值

        return soft_mask


if __name__ == "__main__":
    torch.manual_seed(1037)
    x = torch.randn(2, 3, 4)  # (batch_size, num_features, patch_size)
    channel_mask_generator = LinearChannelMasker(new_patch_size=4, num_features=3)
    resample_matrix = channel_mask_generator(x)
    print(resample_matrix)

    # 测试 GATChannelMasker
    # 创建一个简单的输入张量
    batch_size = 2
    num_channels = 3
    node_feature_dim = 4
    x_nodes = torch.randn(batch_size, num_channels, node_feature_dim)

    # 初始化 GATChannelMasker
    masker = GATChannelMasker(node_feature_dim, num_channels)

    # 前向传播
    soft_mask = masker(x_nodes)
    print(soft_mask)
