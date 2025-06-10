import torch
import torch.nn as nn

x = torch.randn(2, 2, 4, 4)  # (batch_size, num_features, patch_num, d_model * 2)

print(x[:, 0, :, :].shape)

# print(nn.Flatten(start_dim=2)(x).shape)
