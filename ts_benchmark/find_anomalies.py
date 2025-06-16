import argparse
import tomllib
from typing import Any, Dict

import numpy as np
import pandas as pd
from baselines.catch.catch_pipeline import catch_find_anomalies, catch_score_anomalies
from tools.tools import set_seed

from ts_benchmark.evaluation.metrics.anomaly_score_metrics import auc_roc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="配置文件路径",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="数据集文件路径",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(1037)

    # 加载配置文件
    catch_config: Dict[Any, Any]
    with open(args.config, "rb") as f:
        catch_config = tomllib.load(f)

    df = pd.read_csv(args.dataset)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1].to_numpy()

    scores = catch_score_anomalies(data=data.values, config=catch_config)
    # np.savetxt("scores.csv", scores, delimiter=",", fmt="%.6f")
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
