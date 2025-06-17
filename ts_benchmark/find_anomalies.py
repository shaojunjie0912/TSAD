import argparse
import tomllib
from typing import Any, Dict

import numpy as np
import pandas as pd
from baselines.catch.catch_pipeline import catch_find_anomalies
from evaluation.metrics.anomaly_detection_metrics_label import affiliation_f
from tools.tools import set_seed


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

    predictions = catch_find_anomalies(data=data.values, config=catch_config)
    print(affiliation_f(labels, predictions))
    # np.savetxt("predictions.csv", predictions, delimiter=",", fmt="%d")

    # plot_anomaly_labels(
    #     data=data,
    #     predictions=predictions,
    #     results_dir="results",
    #     dataset_name="common_dataset",
    #     algorithm_name="CATCH",
    # )

    print("----------------- 🆗 -----------------")
