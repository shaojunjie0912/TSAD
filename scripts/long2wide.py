import glob
import os

import pandas as pd


def process_data_df(input_path: str, output_path: str, nrows=None):
    data = pd.read_csv(input_path)
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]

    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    df.to_csv(output_path, index=False)


def batch_convert_directory(input_dir, output_dir, suffix="_wide"):
    """
    批量转换一个目录下的所有CSV文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出将保存到：{output_dir}\n")
    search_pattern = os.path.join(input_dir, "*.csv")
    csv_files = glob.glob(search_pattern)
    if not csv_files:
        print(f"在 '{input_dir}' 中未找到CSV文件。")
        return
    print(f"找到 {len(csv_files)} 个CSV文件待处理...")
    for file_path in csv_files:
        print(f"正在处理 '{os.path.basename(file_path)}'...")
        base_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_filename = f"{file_name_without_ext}{suffix}.csv"
        output_filepath = os.path.join(output_dir, output_filename)
        process_data_df(file_path, output_filepath)
    print("\n批量转换完成！")


if __name__ == "__main__":
    batch_convert_directory("long_data", "wide_data", suffix="")
