import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#  我已经支持结果路径+数据集名称+算法名称, 不同模型参数咋办?

# 如何保存不同模型参数和对应的结果图?
# 1. 是否需要增加一个形参表示模型参数如果修改了参数是否应该保存到不同的目录?
#


def plot_anomaly_scores(
    data: pd.DataFrame, scores: np.ndarray, results_dir: str, algorithm_name: str, dataset_name: str
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
    ax = axes[0]
    ax.plot(scores, color=score_color, linewidth=1.5, label="Anomaly Score")
    ax.set_ylabel("Anomaly Score")
    ax.legend(loc="upper right")

    # 绘制每个变量的时序图
    for i, col in enumerate(data.columns):
        ax = axes[i + 1]
        ax.plot(data[col], color=original_color, linewidth=1.5, label=col)
        ax.set_ylabel(col)
        ax.legend(loc="upper right")

    plt.xlabel("Time Step")
    plt.tight_layout()
    path = f"{results_dir}/{dataset_name}/{algorithm_name}/anomaly_scores"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"{path}/anomaly_scores.png", dpi=400, bbox_inches="tight")


def plot_anomaly_indices(
    data: pd.DataFrame,
    predictions: Dict[float, np.ndarray],
    results_dir: str,
    dataset_name: str,
    algorithm_name: str,
):
    num_features = data.shape[1]
    original_color = "#4a6fa5"  # 深灰蓝，低调高级
    error_color = "#e76f51"  # 柔和珊瑚红

    for ratio, pred in predictions.items():
        fig, axes = plt.subplots(num_features, 1, figsize=(15, 3 * num_features), sharex=True)
        fig.suptitle(f"Anomalies with ratio: {ratio}")

        # 获取异常时刻的索引
        anomaly_indices = np.where(pred == 1)[0]

        # 绘制每个变量的时序图
        for i, col in enumerate(data.columns):
            ax = axes[i]
            # 使用data的index作为x轴
            ax.plot(data.index, data[col], label=col, color=original_color, linewidth=1.5)

            # 在每个异常时刻画一条垂直的红线
            for idx in anomaly_indices:
                ax.axvline(
                    x=data.index[idx], color=error_color, alpha=0.3, linestyle="-", linewidth=1
                )

            ax.set_ylabel(col)
            ax.legend(loc="upper right")

        plt.xlabel("Time")
        plt.tight_layout()

        path = f"{results_dir}/{dataset_name}/{algorithm_name}/anomaly_indices"
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(f"{path}/{ratio}.png", dpi=400, bbox_inches="tight")
        plt.close()
