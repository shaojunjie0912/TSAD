import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            node_feature_dim (int): 每个通道节点输入的特征维度 (来自 WaveletPatcher, 即 patch_size)
            num_channels (int): 通道数量 (即图中的节点数量)
            num_gat_heads (int): GAT 中的注意力头数
            gat_head_dim (int): 每个GAT注意力头的维度
            dropout_rate (float): Attention权重的Dropout比率
        """
        super().__init__()
        # 更新通道数：num_features 是每个尺度的通道数，乘以 (L+1) 扩展通道数
        self.num_channels = num_features  # 注意：num_features 为 (L+1) * num_features
        self.num_gat_heads = num_gat_heads
        self.gat_head_dim = gat_head_dim
        self.inner_dim = self.gat_head_dim * self.num_gat_heads  # Q, K 的总维度

        # 线性变换层，用于计算 Query 和 Key
        self.to_q = nn.Linear(node_feature_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(node_feature_dim, self.inner_dim, bias=False)

        self.attention_dropout = nn.Dropout(dropout_rate)

        # 初始化权重，使得初始的注意力分布是均匀的
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
        q_channels = self.to_q(x_nodes)
        k_channels = self.to_k(x_nodes)

        # 重塑以支持多头注意力
        q_channels = einops.rearrange(
            q_channels, "b n (h d) -> b h n d", h=self.num_gat_heads, d=self.gat_head_dim
        )
        k_channels = einops.rearrange(
            k_channels, "b n (h d) -> b h n d", h=self.num_gat_heads, d=self.gat_head_dim
        )

        # 计算缩放点积注意力原始分数 (相似度)
        scale = math.sqrt(self.gat_head_dim)
        channel_sim = torch.einsum("b h i d, b h j d -> b h i j", q_channels, k_channels) / scale

        # 应用 Softmax 得到注意力权重
        channel_att_weights = F.softmax(channel_sim, dim=-1)
        channel_att_weights = self.attention_dropout(channel_att_weights)

        # 平均多个头的注意力权重得到最终的软掩码
        soft_mask = channel_att_weights.mean(dim=1)

        return soft_mask


if __name__ == "__main__":
    torch.manual_seed(1037)

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
