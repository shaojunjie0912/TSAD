import torch

# TODO: 反正就是一些数学公式
# 聚类损失: 应用「对比学习」公式，拉近正样本对，推远负样本对
# 正则化损失: 鼓励掩码保持「稀疏性」，防止所有通道都高度相关


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
