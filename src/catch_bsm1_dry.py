import os
import tomllib
from typing import Any, Dict

import pandas as pd

from CATCH.catch_pipeline import catch_find_anomalies, catch_score_anomalies
from tools.plot import plot_anomaly_labels, plot_anomaly_scores


def load_config(path: str) -> Dict[str, Any]:
    if path == "":
        raise ValueError("Please specify the path of config file!")
    else:
        with open(path, "rb") as f:
            return tomllib.load(f)


# TODO: 如果要统一不同模型不同数据集的 pipeline, 则需要:
# 1. 配置文件路径
# 2. 数据集路径
# 3. ??? 目标列名? fillna

if __name__ == "__main__":
    # 加载配置文件
    catch_config = load_config("configs/bsm1_dry/CATCH/catch.toml")
    # 加载数据
    df = pd.read_csv("datasets/bsm1_dry/inputs.csv", index_col=0, parse_dates=[0])

    data = df.loc[
        :,
        # ["Ss", "Xi", "Xs", "Xbh", "Snh", "Snd", "Xnd", "Q"],
        ["Ss", "Xi", "Xs", "Xbh"],
    ]

    scores = catch_score_anomalies(data=data.values, config=catch_config)
    plot_anomaly_scores(
        data=data,
        scores=scores,
        results_dir="results",
        dataset_name="bsm1_dry",
        algorithm_name="CATCH",
    )

    predictions = catch_find_anomalies(data=data.values, config=catch_config)

    plot_anomaly_labels(
        data=data,
        predictions=predictions,
        results_dir="results",
        dataset_name="bsm1_dry",
        algorithm_name="CATCH",
    )

    print("----------------- 🆗 -----------------")
