from datetime import datetime

import pandas as pd
from anomaly_injector import AnomalyInjector, RandomAnomConfig


def load_bsm1_data(txt_file: str, start_datetime_str: str) -> pd.DataFrame:
    # BSM1进水文件中的列名顺序 (根据BSM1技术报告)
    # 所有: t Si Ss Xi Xs Xbh Xba Xp So Sno Snh Snd Xnd Salk Q
    # 变化: Ss Xi Xs Xbh Snh Snd Xnd Q
    try:
        df = pd.read_csv(txt_file, sep="\t")
    except Exception as e:
        print(f"读取文件时出错 {txt_file}: {e}")
        print("请确保文件路径正确，并且文件内容是以空格或制表符分隔的。")
        return pd.DataFrame()

    # 将时间（天）转换为datetime对象
    # BSM1数据通常以15分钟为间隔，1天 = 24 * 4 = 96个采样点
    start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")

    # Time_days 相对于 0, 采样间隔是 15 分钟
    # 第一个时间点通常是 0，第二个是 0 + 15/(24*60) 等
    num_samples = len(df)
    time_index = pd.date_range(start=start_datetime, periods=num_samples, freq="15min")
    df.index = time_index

    df.drop(columns=["t"], inplace=True)  # 移除原始天数时间列

    print(f"✅ 成功加载 {len(df)} 条数据, 时间范围: {df.index.min()} ~ {df.index.max()}")
    return df


if __name__ == "__main__":
    df_raw = load_bsm1_data(
        txt_file="datasets/bsm1/txt/Inf_dry_2006.txt", start_datetime_str="2006-01-01 00:00:00"
    )

    cfg = RandomAnomConfig(
        anomaly_rate=0.05,
        seed=1037,
        target_cols=["Ss", "Xi", "Xs", "Xbh", "Snh", "Snd", "Xnd", "Q"],
        interval_len_range=(4, 12),  # 1.5 h ~ 6 h
        fixed_value_strategy=("percentile", 0.95),
    )

    injector = AnomalyInjector(clamp_min=0)

    df_anom, labels_detailed, labels_simple = injector.inject_random_anomalies(df_raw, config=cfg)

    df_anom.to_csv("datasets/bsm1_dry/inputs.csv")
    labels_detailed.to_csv("datasets/bsm1_dry/labels_detailed.csv")
    labels_simple.to_csv("datasets/bsm1_dry/labels.csv", header=False)

    print("✅ 完成异常注入和标签生成")
