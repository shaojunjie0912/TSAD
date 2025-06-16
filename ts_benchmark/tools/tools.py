import os
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


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


def plot_anomaly_scores(
    data: pd.DataFrame,
    scores: np.ndarray,
    results_dir: str,
    dataset_name: str,
    algorithm_name: str,
):
    """
    绘制异常评分和原始数据

    Args:
        data (pd.DataFrame): 原始数据
        scores (np.ndarray): 异常评分
    """
    num_features = data.shape[1]
    original_color = "#4a6fa5"  # 深灰蓝，低调高级
    score_color = "#2a9d8f"  # 绿松石色

    fig, axes = plt.subplots(num_features + 1, 1, figsize=(15, 3 * (num_features + 1)), sharex=True)
    fig.suptitle("Anomaly Scores and Original Data")

    # 绘制异常评分
    ax = axes[-1]
    ax.plot(data.index, scores, color=score_color, linewidth=1.5, label="Anomaly Score")
    ax.set_ylabel("Anomaly Score")
    ax.legend(loc="upper right")

    # 绘制每个变量的时序图
    for i, col in enumerate(data.columns):
        ax = axes[i]
        ax.plot(data.index, data[col], color=original_color, linewidth=1.5, label=col)
        ax.set_ylabel(col)
        ax.legend(loc="upper right")

    plt.xlabel("Time")
    plt.tight_layout()
    path = f"{results_dir}/{dataset_name}/{algorithm_name}/anomaly_scores"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"{path}/anomaly_scores.png", dpi=400, bbox_inches="tight")


def plot_anomaly_labels(
    data: pd.DataFrame,
    predictions: np.ndarray,
    results_dir: str,
    dataset_name: str,
    algorithm_name: str,
):
    num_features = data.shape[1]
    original_color = "#4a6fa5"  # 深灰蓝，低调高级
    error_color = "#e76f51"  # 柔和珊瑚红

    fig, axes = plt.subplots(num_features, 1, figsize=(15, 3 * num_features), sharex=True)
    fig.suptitle(f"Anomalies")

    # 获取异常时刻的索引
    anomaly_labels = np.where(predictions == 1)[0]

    # 绘制每个变量的时序图
    for i, col in enumerate(data.columns):
        ax = axes[i]
        # 使用data的index作为x轴
        ax.plot(data.index, data[col], label=col, color=original_color, linewidth=1.5)

        # 在每个异常时刻画一条垂直的红线
        for idx in anomaly_labels:
            ax.axvline(x=data.index[idx], color=error_color, alpha=0.3, linestyle="-", linewidth=1)

        ax.set_ylabel(col)
        ax.legend(loc="upper right")

    plt.xlabel("Time")
    plt.tight_layout()

    path = f"{results_dir}/{dataset_name}/{algorithm_name}/anomaly_labels"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"{path}/anomaly_labels.png", dpi=400, bbox_inches="tight")
    plt.close()
