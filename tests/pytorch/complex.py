import torch

# # 复数
# x = torch.tensor([1 - 3j])
# y = torch.tensor([2 + 4j])


# # 计算相位差
# phase_diff = torch.angle(x) - torch.angle(y)
# print("Phase difference:", phase_diff)

# print(x**2)

# # 计算幅度差
# magnitude_diff = torch.abs(x) - torch.abs(y)
# print("Magnitude difference:", magnitude_diff)

# # 计算复数差
# complex_diff = x - y
# print("Complex difference:", complex_diff)

# 2 * 2 复数矩阵
x = torch.tensor(
    [
        [1 - 3j, 2 + 4j],
        [3 - 5j, 4 + 6j],
    ]
)

x_real = x.real
x_imag = x.imag

print(torch.cat((x_real, x_imag), dim=-1))
