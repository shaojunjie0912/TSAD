import pandas as pd
import numpy as np

# 创建一个三行两列的DataFrame
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

print(df.iloc[:1, :].reset_index(drop=True))
print("\n")
print(df.iloc[1:, :].reset_index(drop=True))
print("\n")
print(df)
