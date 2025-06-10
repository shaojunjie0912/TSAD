import torch

print(torch.cuda.is_available())

# num_samples * seq_len
x = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0],
    ]
)

y = torch.tensor(
    [
        [11.0, 12.0, 13.0],
        [13.0, 14.0, 15.0],
    ]
)

print(torch.cat((x, y), dim=0))
