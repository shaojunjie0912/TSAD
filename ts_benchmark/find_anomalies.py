import argparse
import tomllib
from typing import Any, Dict

import numpy as np
import pandas as pd
from baselines.swift.swift_pipeline import swift_find_anomalies
from evaluation.metrics.anomaly_detection_metrics_label import affiliation_f
from tools.tools import parse_args, set_seed

if __name__ == "__main__":
    args = parse_args()
    set_seed(1037)

    # 加载配置文件
    swift_config: Dict[Any, Any]
    with open(args.config, "rb") as f:
        swift_config = tomllib.load(f)

    df = pd.read_csv(args.dataset)

    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1].to_numpy()

    predictions = swift_find_anomalies(data=data.values, config=swift_config)
    print(f"Affiliation F1 Score: {affiliation_f(labels, predictions)}")
    # np.savetxt("predictions.csv", predictions, delimiter=",", fmt="%d")

    # plot_anomaly_labels(
    #     data=data,
    #     predictions=predictions,
    #     results_dir="results",
    #     dataset_name="common_dataset",
    #     algorithm_name="SWIFT",
    # )

    print("----------------- 🆗 -----------------")
