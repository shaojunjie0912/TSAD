import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenHead(nn.Module):
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
            input_dim (int): 输入特征维度 (d_model * 2 * patch_num)
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
                # NOTE: flatten 从倒数第 2 维 patch_num, d_model * 2 开始展平
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
            x (torch.Tensor): (batch_size, num_features, patch_num, d_model * 2)

        Returns:
            torch.Tensor: (batch_size, num_features, seq_len)
        """
        if self.individual:
            x_out = []
            for i in range(self.num_features):
                z = self.flattens[i](x[:, i, :, :])  # z: (batch_size, d_model * 2 * patch_num)
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
