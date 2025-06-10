import torch
import torch.nn as nn

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x_hat = torch.tensor([[1.5, 1.5], [2.5, 4.0]])

mse_loss = nn.MSELoss(reduction="none")

print(mse_loss(x, x_hat))
print(torch.mean(mse_loss(x, x_hat), dim=-1))
