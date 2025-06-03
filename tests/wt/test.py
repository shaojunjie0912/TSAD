from typing import List, Tuple

import ptwt
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    # (batch_size, num_features, seq_len)
    # 写一个 (1, 3, 5) 的 tensor
    x = torch.tensor(
        [
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        ],
        dtype=torch.float32,
    )
