#!/usr/bin/env python3

"""
SWIFT 参数自动调优脚本
每个配置完成后立即保存最佳结果，支持断点续传
使用 tomli 读取 TOML，tomli-w 写入 TOML
"""

import copy
import itertools
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tomli
import tomli_w
from baselines.swift.swift_pipeline import swift_find_anomalies
from evaluation.metrics.anomaly_detection_metrics_label import affiliation_f
from tools.tools import set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, "rb") as f:
        return tomli.load(f)


def update_config(base_config: Dict[str, Any], param_updates: Dict[str, Any]) -> Dict[str, Any]:
    """更新配置参数"""
    config = copy.deepcopy(base_config)

    for key_path, value in param_updates.items():
        keys = key_path.split(".")
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value

    return config


def evaluate_config(data: np.ndarray, labels: np.ndarray, config: Dict[str, Any]) -> float | None:
    """评估单个配置的性能"""
    try:
        predictions = swift_find_anomalies(data=data, config=config)
        score = affiliation_f(labels, predictions)
        return score
    except Exception as e:
        print(f"配置评估失败: {e}")
        return None


def save_progress(progress_file: str, results: List[Dict], best_params: Dict, best_score: float):
    """保存搜索进度"""
    progress_data = {
        "results": results,
        "best_params": best_params,
        "best_score": best_score,
        "completed_configs": len(results),
    }
    temp_file = progress_file + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(progress_data, f, indent=2)
    # 写入成功后，再重命名
    os.replace(temp_file, progress_file)


def load_progress(progress_file: str) -> tuple[List[Dict], Dict, float, int]:
    """加载搜索进度"""
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress_data = json.load(f)
        return (
            progress_data.get("results", []),
            progress_data.get("best_params", {}),
            progress_data.get("best_score", 0.0),
            progress_data.get("completed_configs", 0),
        )
    return [], {}, 0.0, 0


def save_optimal_config(base_config: Dict[str, Any], best_params: Dict[str, Any], output_path: str):
    """保存优化后的配置到TOML文件"""
    if not best_params:
        return

    optimal_config = update_config(base_config, best_params)

    # 添加注释到配置顶部
    optimal_config["_comment"] = "自动调优生成的最佳配置"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 使用 tomli_w 写入 TOML 文件
    with open(output_path, "wb") as f:
        tomli_w.dump(optimal_config, f)

    print(f"💾 配置已保存为 TOML 格式: {output_path}")


def quick_search_params(
    data: np.ndarray,
    labels: np.ndarray,
    base_config: Dict[str, Any],
    output_path: str,
    progress_file: str = "quick_tune_progress.json",
) -> tuple[Dict[str, Any], float]:
    """快速搜索重要参数（每个配置都实时保存）"""

    # 重要参数的搜索空间
    param_grid = {
        "anomaly_detection.scale_score_lambda": [0.15, 0.2, 0.25, 0.3],
        "anomaly_detection.anomaly_ratio": [2.0, 2.5, 3.0],
        "anomaly_detection.aggregation_method": ["max", "weighted_max"],
        "model.CFM.dropout": [0.1, 0.15, 0.2],
        "training.learning_rate": [0.0005, 0.001],
    }

    total_configs = int(np.prod([len(v) for v in param_grid.values()]))

    # 尝试加载之前的进度
    results, best_params, best_score, completed_configs = load_progress(progress_file)

    if completed_configs > 0:
        print(f"🔄 检测到之前的进度，已完成 {completed_configs}/{total_configs} 个配置")
        print(f"📊 当前最佳分数: {best_score:.4f}")
    else:
        print(f"🆕 开始快速搜索，总计配置数: {total_configs}")

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))

    for i, combination in enumerate(all_combinations):
        # 跳过已完成的配置
        if i < completed_configs:
            continue

        param_updates = dict(zip(param_names, combination))
        config = update_config(base_config, param_updates)
        score = evaluate_config(data, labels, config)

        results.append({"params": param_updates, "score": score})

        if score is not None and score > best_score:
            best_score = score
            best_params = param_updates
            print(f"🎉 发现更好配置! F1={score:.4f}")
            # 立即保存新的最佳配置
            save_optimal_config(base_config, best_params, output_path)

        print(f"配置 {i+1}/{total_configs}: F1={score:.4f} | 当前最佳: {best_score:.4f}")

        # 每次都保存进度
        save_progress(progress_file, results, best_params, best_score)

    return best_params, best_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="基础配置文件路径")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="数据集文件路径")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="configs/find_anomalies/ASD1/swift_optimized.toml",
        help="输出配置文件路径",
    )
    parser.add_argument("--resume", action="store_true", help="从之前的进度继续搜索")

    args = parser.parse_args()

    set_seed(1037)

    # 加载数据和配置
    print("🚀 开始SWIFT参数自动调优...")

    base_config = load_config(args.config)
    df = pd.read_csv(args.dataset)
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].to_numpy()

    # print(f"数据形状: {data.shape}, 异常点数量: {np.sum(labels)}")

    # 设置进度文件名
    progress_file = "tune_params_progress.json"

    if not args.resume and os.path.exists(progress_file):
        print(f"⚠️ 发现之前的进度文件 {progress_file}")
        response = input("是否继续之前的搜索? (y/n): ").lower().strip()
        if response != "y":
            os.remove(progress_file)
            print("已删除之前的进度文件，开始新的搜索")

    print("执行快速参数搜索...")
    best_params, best_score = quick_search_params(data, labels, base_config, args.output, progress_file)

    print(f"\n🎉 搜索完成!")
    print(f"最佳F1分数: {best_score:.4f}")
    print(f"最佳参数: {best_params}")

    if best_params:
        print(f"✅ 最终配置已保存到: {args.output}")
        # 清理进度文件
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"🧹 已清理进度文件: {progress_file}")
    else:
        print("❌ 未找到更好的参数配置")
