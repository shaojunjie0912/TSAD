import numpy as np

x = [1, 2, 3]
h = [1, 0.5]

y = np.convolve(x, h, mode="full")  # 'full' 会输出完整的卷积结果
print(y)  # 输出: [1. 2.5 4. 1.5]
