import argparse
import random
import tomllib
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from baselines.swift.swift_pipeline import swift_score_anomalies
from sklearn.metrics import roc_auc_score


def auc_roc(y_true: np.ndarray, y_scores: np.ndarray):
    """计算 ROC 曲线下面积 (AUC-ROC)

    Args:
        y_true (np.ndarray): 真实标签
        y_scores (np.ndarray): 预测得分

    Returns:
        float: AUC-ROC 值
    """
    return roc_auc_score(y_true, y_scores)


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


def set_seed(seed: int = 1037):
    random.seed(seed)  # python 内置
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch-cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # pytorch-gpu
        torch.cuda.manual_seed_all(seed)  # pytorch-all-gpu
        # 确保 cudnn 的确定性，但这可能会牺牲一些性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

    scores = swift_score_anomalies(all_data=data.values, config=swift_config)
    print(auc_roc(labels, scores))

    print("----------------- ✅ -----------------")
