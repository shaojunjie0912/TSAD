import torch

# 测试 stack

x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)

stack1 = torch.stack([x, x], dim=0)  # 在第 0 维上拼接
print(stack1.shape)  # torch.Size([2, 2, 3])
print(stack1)

stack2 = torch.stack([x, x], dim=1)  # 在第 1 维上拼接
print(stack2.shape)  # torch.Size([2, 2, 3])
print(stack2)

stack3 = torch.stack([x, x], dim=2)  # 在第 2 维上拼接
print(stack3.shape)  # torch.Size([2, 3, 2])
print(stack3)
