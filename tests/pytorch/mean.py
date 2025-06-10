import torch

# 原始 x shape(2, 2)
x = torch.tensor(
    [
        [1, 2],
        [3, 4],
    ],
    dtype=torch.float32,
)

# dim = 0 不保留维度得到 shape(2,)
print(x.mean(dim=0))

# dim = 0 保留维度得到 shape(1, 2)
print(x.mean(dim=0, keepdim=True))
