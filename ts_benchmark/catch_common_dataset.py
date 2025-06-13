import os
import tomllib
from typing import Any, Dict

import numpy as np
import pandas as pd
from baselines.catch.catch_pipeline import catch_find_anomalies, catch_score_anomalies
from evaluation.metrics.classification_metrics_score import auc_roc
from tools.plot import plot_anomaly_labels, plot_anomaly_scores


def load_config(path: str) -> Dict[str, Any]:
    if path == "":
        raise ValueError("Please specify the path of config file!")
    else:
        with open(path, "rb") as f:
            return tomllib.load(f)


if __name__ == "__main__":
    # 加载配置文件
    catch_config = load_config("configs/ASD1/catch.toml")
    # 加载数据 (最后一列 label 是标签, 前面所有列是各个通道数据)
    df = pd.read_csv("datasets/ASD_dataset_1.csv")

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1].to_numpy()
    scores = catch_score_anomalies(data=data.values, config=catch_config)
    np.savetxt("scores.csv", scores, delimiter=",", fmt="%.6f")
    print(auc_roc(labels, scores))
    # plot_anomaly_scores(
    #     data=data,
    #     scores=scores,
    #     results_dir="results",
    #     dataset_name="common_dataset",
    #     algorithm_name="CATCH",
    # )

    # predictions = catch_find_anomalies(data=data.values, config=catch_config)
    # np.savetxt("predictions.csv", predictions, delimiter=",", fmt="%d")

    # plot_anomaly_labels(
    #     data=data,
    #     predictions=predictions,
    #     results_dir="results",
    #     dataset_name="common_dataset",
    #     algorithm_name="CATCH",
    # )

    print("----------------- 🆗 -----------------")
