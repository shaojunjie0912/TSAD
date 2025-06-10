import numpy as np
import pandas as pd

# 以 datetime 为 index
df = pd.DataFrame(
    {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
    },
    index=pd.date_range(start="2022-01-01", periods=3),
)

time_index_str = "2022-01-01"
time_index = pd.to_datetime(time_index_str)

time_indices_str = ["2022-01-01", "2022-01-02"]

time_indices = pd.to_datetime(time_indices_str)

print("原始 DataFrame:")
print(df)

selected_df_A = df.loc[time_indices, ["A"]]
print("\n选择的 DataFrame 部分:")
print(selected_df_A)

print("\n与标量相乘:")
print(selected_df_A * 2)

# 报错行：
# print(df.loc[time_indices, ["A"]] * [2, 3])

# 解决方案 1: 使用 NumPy 数组并 reshape
multiplier_array = np.array([2, 3]).reshape(-1, 1)  # -1 表示自动计算该维度, 1 表示1列
print("\n使用 reshape后的 NumPy 数组相乘:")
print(selected_df_A * multiplier_array)
