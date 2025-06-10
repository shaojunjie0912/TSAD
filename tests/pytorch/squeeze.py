import torch

x = torch.tensor([[1], [2], [4]])  # (3, 1)
torch.squeeze(x)  # (3,)

# print(x.unsqueeze(0))
