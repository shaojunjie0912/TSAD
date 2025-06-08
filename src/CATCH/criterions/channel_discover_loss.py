import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCCDLoss(nn.Module):
    """
    自适应通道关联发现损失 (Adaptive Channel Correlation Discovering Loss)。
    结合了通道注意力对齐损失和掩码正则化损失，用于指导 GATChannelMasker (软掩码)
    和 ChannelMaskedTransformer 的学习。
    """

    def __init__(
        self,
        alignment_temperature: float = 0.1,
        regularization_lambda: float = 0.1,
        alignment_lambda: float = 1.0,
    ):  # 新增对齐损失的权重
        """
        Args:
            alignment_temperature (float): 用于注意力对齐损失中softmax的温度参数。
            regularization_lambda (float): 掩码正则化损失的权重系数。
            alignment_lambda (float): 通道注意力对齐损失的权重系数。
        """
        super().__init__()
        self.alignment_temperature = alignment_temperature
        self.regularization_lambda = regularization_lambda
        self.alignment_lambda = alignment_lambda

    def forward(
        self, cfm_attention_logits: torch.Tensor, soft_channel_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算总的CCD损失。

        Args:
            cfm_attention_logits (torch.Tensor):
                来自 ChannelMaskedTransformer (CMT) 注意力模块的原始相似度得分 (Q*K^T)。
                这些应该是未经softmax和未经掩码的原始分数。
                期望形状: (B_eff, num_cfm_heads, N, N) or (B_eff, N, N) if pre-averaged.
            soft_channel_mask (torch.Tensor):
                由 GATChannelMasker 生成的软掩码。
                期望形状: (B_eff, N, N)。其每一行应近似一个概率分布 (和为1)。

        Returns:
            torch.Tensor: 计算得到的总CCD损失 (标量)。
        """
        b_eff, n_channels, _ = soft_channel_mask.shape

        # --- 1. 通道注意力对齐损失 ---
        # (鼓励CMT的注意力分布 P_cfm 接近 GAT软掩码提供的目标分布 M_soft)

        # 如果 cfm_attention_logits 有头维度，则先平均
        if cfm_attention_logits.ndim == 4:  # (B_eff, H_cfm, N, N)
            cfm_att_logits_avg_heads = cfm_attention_logits.mean(dim=1)  # (B_eff, N, N)
        elif cfm_attention_logits.ndim == 3:  # (B_eff, N, N)
            cfm_att_logits_avg_heads = cfm_attention_logits
        else:
            raise ValueError(
                f"Unsupported cfm_attention_logits shape: {cfm_attention_logits.shape}"
            )

        # 将CMT的logits转换为对数概率 (log P_cfm(j|i))
        # 除以温度参数可以锐化或平滑分布
        log_probs_cfm_attention = F.log_softmax(
            cfm_att_logits_avg_heads / self.alignment_temperature,
            dim=-1,  # 对每个查询通道i，在所有可能的键通道j上进行softmax
        )  # (B_eff, N, N)

        # soft_channel_mask (M_soft) 作为目标概率分布 P_gat(j|i)
        # 计算交叉熵: - sum_{i,j} M_soft[i,j] * log_P_cfm[i,j]
        # (sum over j first, then mean over i, then mean over B_eff)
        alignment_loss = -(soft_channel_mask * log_probs_cfm_attention).sum(
            dim=-1
        )  # Sum over j -> (B_eff, N)
        alignment_loss = alignment_loss.mean()  # Mean over B_eff and N (query channels)

        # --- 2. 掩码正则化损失 ---
        # (鼓励 GATChannelMasker 生成的软掩码 M_soft 接近单位矩阵 I)
        identity_matrix = torch.eye(
            n_channels, device=soft_channel_mask.device, dtype=soft_channel_mask.dtype
        )
        identity_matrix = identity_matrix.unsqueeze(0).expand_as(soft_channel_mask)  # (B_eff, N, N)

        # 使用 Frobenius 范数: (1/N) * ||I - M_soft||_F
        # torch.norm(..., p='fro', dim=(-2,-1)) 计算每个batch元素的Frobenius范数
        l2_diff_from_identity = torch.norm(
            identity_matrix - soft_channel_mask, p="fro", dim=(-2, -1)
        )  # (B_eff,)
        mask_regularization_loss = (l2_diff_from_identity / n_channels).mean()  # Mean over B_eff

        # --- 总损失 ---
        total_ccd_loss = (
            self.alignment_lambda * alignment_loss
            + self.regularization_lambda * mask_regularization_loss
        )

        return total_ccd_loss
