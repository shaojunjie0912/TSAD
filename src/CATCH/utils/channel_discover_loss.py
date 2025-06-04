import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: 反正就是一些数学公式
# 聚类损失: 应用「对比学习」公式，拉近正样本对，推远负样本对
# 正则化损失: 鼓励掩码保持「稀疏性」，防止所有通道都高度相关


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


class DynamicalContrastiveLoss(torch.nn.Module):
    """动态对比损失函数

    用于发现和优化时间序列中的通道间依赖关系,
    这个损失函数通过对比学习的方式引导模型学习更有意义的通道关系表示.
    """

    def __init__(self, temperature=0.5, regular_lambda=0.3):
        """_summary_

        Args:
            temperature (float, optional): 控制相似度分布的"锐度", 较低值使模型更确定地区分相似和不相似通道.
            regular_lambda (float, optional): 正则化系数, 控制稀疏性约束的强度, 平衡聚类目标和结构约束.
        """
        super().__init__()
        self.temperature = temperature
        self.regular_lambda = regular_lambda

    def _stable_scores(self, scores):
        max_scores = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        stable_scores = scores - max_scores
        return stable_scores

    def forward(
        self,
        sims: torch.Tensor,
        norm_sims: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            sims (torch.Tensor): Q * K^T 相似度矩阵 shape: (b, h, n, n)
            channel_mask (torch.Tensor): 注意力掩码矩阵 shape: (b, h, n, n)
            norm_sims (torch.Tensor): ||Q|| * ||K|| F-范数矩阵 shape: (b, h, n, n)

        Returns:
            torch.Tensor: _description_
        """
        batch_size = sims.shape[0]  # NOTE: 其实是 batch_size * patch_num
        num_features = sims.shape[-1]

        # -------------------------------------
        # ------------ 聚类损失 ------------------
        # -------------------------------------

        # NOTE: 余弦相似度对于每个头的平均值
        cos_sims = (sims / norm_sims).mean(dim=1)  # shape: (b, n, n)
        all_scores = torch.exp(cos_sims / self.temperature)
        pos_scores = cos_sims * channel_mask

        # 添加一个小的epsilon值防止除零和log(0)
        epsilon = 1e-10

        # 确保分母不为零
        denominator = all_scores.sum(dim=-1).clamp(min=epsilon)
        # 确保分子不为零且为正数
        numerator = pos_scores.sum(dim=-1).clamp(min=epsilon)

        # 计算损失
        clustering_loss = -torch.log(numerator / denominator)

        # -------------------------------------
        # ------------ 正则化损失 ------------------
        # -------------------------------------

        eye = (
            torch.eye(channel_mask.shape[-1], device=channel_mask.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        regular_loss = (
            1
            / (num_features * (num_features - 1))
            * torch.norm(
                eye.reshape(batch_size, -1) - channel_mask.reshape((batch_size, -1)), p=1, dim=-1
            )
        )

        loss = clustering_loss.mean(1) + self.regular_lambda * regular_loss

        mean_loss = loss.mean()
        return mean_loss
