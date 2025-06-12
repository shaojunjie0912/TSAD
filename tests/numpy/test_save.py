import numpy as np

a = np.array([1, 2, 3, 4, 5])

# 保存为csv
np.savetxt("a.csv", a, delimiter=",", fmt="%.2f")
