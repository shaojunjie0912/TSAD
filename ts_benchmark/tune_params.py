#!/usr/bin/env python3

"""
使用 Optuna 对 SWIFT 模型进行高效的超参数优化（HPO）。

该脚本替代了原有的基于网格搜索的 tune_params.py，具备以下优点：
1.  **高效搜索**: 使用贝叶斯优化算法 (TPE)，而非低效的网格搜索。
2.  **断点续传**: 自动将进度保存到SQLite数据库，可随时中断和恢复。
3.  **实时保存**: 每当发现更优的配置时，都会立即将其保存到 TOML 文件。
4.  **清晰的依赖处理**: 优雅地处理了 d_model >= d_cf 这类参数依赖。
5.  **支持并行**: （需稍作配置）Optuna 支持多进程并行调优。

使用方法:
    python tune_with_optuna.py \
        -c configs/find_anomalies/ASD1/swift.toml \
        -d ./datasets/processed/ASD/train_1.csv \
        -o configs/find_anomalies/ASD1/swift_optuna_best.toml \
        --n-trials 100 \
        --study-name "swift-asd-tuning"
"""

import argparse
import copy
import os
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
import tomli
import tomli_w
from baselines.swift.swift_pipeline import swift_find_anomalies
from evaluation.metrics.anomaly_detection_metrics_label import affiliation_f
from tools.tools import set_seed

# --------------------------------------------------------------------------- #
#                             辅助函数 (来自原脚本)                             #
# --------------------------------------------------------------------------- #


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 TOML 配置文件。"""
    with open(config_path, "rb") as f:
        return tomli.load(f)


def update_config(base_config: Dict[str, Any], param_updates: Dict[str, Any]) -> Dict[str, Any]:
    """根据 Optuna 提供的参数更新配置字典。"""
    config = copy.deepcopy(base_config)
    for key_path, value in param_updates.items():
        keys = key_path.split(".")
        current = config
        for key in keys[:-1]:
            current = current.get(key, {})
        current[keys[-1]] = value
    return config


def save_optimal_config(base_config: Dict[str, Any], best_params: Dict[str, Any], output_path: str):
    """将优化后的配置保存到 TOML 文件。"""
    if not best_params:
        print("🟡 未找到任何有效参数，不保存配置文件。")
        return

    optimal_config = update_config(base_config, best_params)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        tomli_w.dump(optimal_config, f)
    print(f"💾 新的最佳配置已保存到: {output_path}")


# --------------------------------------------------------------------------- #
#                       Optuna 核心目标函数 (Objective)                         #
# --------------------------------------------------------------------------- #


def objective(
    trial: optuna.trial.Trial, base_config: Dict[str, Any], data: np.ndarray, labels: np.ndarray
) -> float:
    """
    Optuna 的目标函数 (第一阶段：核心架构与损失函数调优)。
    用于评估一组对模型性能影响最大的超参数配置。
    """
    param_updates = {}
    param_updates["trial_number"] = trial.number

    # 1. 定义超参数的搜索空间 (第一阶段)
    # -----------------------------------------------------------------------
    # [loss] - 损失函数权重，极为重要
    # -----------------------------------------------------------------------
    param_updates["loss.ccd_loss_lambda"] = trial.suggest_float("loss.ccd_loss_lambda", 1e-4, 1e-1, log=True)
    param_updates["loss.scale_loss_lambda"] = trial.suggest_float("loss.scale_loss_lambda", 0.2, 1.0)
    param_updates["loss.ccd_regular_lambda"] = trial.suggest_float("loss.ccd_regular_lambda", 0.05, 0.5)
    param_updates["loss.ccd_align_lambda"] = trial.suggest_float("loss.ccd_align_lambda", 0.5, 2.0)

    # -----------------------------------------------------------------------
    # [model.FM] - 小波分解层数
    # -----------------------------------------------------------------------
    # 注意: level的改变会影响扩展后的通道数，进而影响GAT模块，这是核心的结构参数
    param_updates["model.FM.level"] = trial.suggest_int("fm.level", 2, 5)

    # -----------------------------------------------------------------------
    # [model.CFM] - Transformer 核心参数
    # -----------------------------------------------------------------------
    param_updates["model.CFM.num_heads"] = trial.suggest_categorical("cfm.num_heads", [2, 4, 8])
    param_updates["model.CFM.attention_dropout"] = trial.suggest_float("cfm.attention_dropout", 0.05, 0.3)

    # 保留原有的关键参数
    d_cf_val = trial.suggest_categorical("cfm.d_cf", [64, 96, 128])
    param_updates["model.CFM.d_cf"] = d_cf_val
    valid_d_model_choices = [val for val in [64, 96, 128] if val >= d_cf_val]
    if not valid_d_model_choices:
        raise optuna.TrialPruned()
    param_updates["model.CFM.d_model"] = trial.suggest_categorical("cfm.d_model", valid_d_model_choices)
    param_updates["model.CFM.num_layers"] = trial.suggest_int("cfm.num_layers", 3, 6)
    param_updates["model.CFM.dropout"] = trial.suggest_float("cfm.dropout", 0.1, 0.25)

    # -----------------------------------------------------------------------
    # [training] - 保留的关键训练参数
    # -----------------------------------------------------------------------
    param_updates["training.learning_rate"] = trial.suggest_float("train.learning_rate", 1e-4, 5e-3, log=True)
    param_updates["training.batch_size"] = trial.suggest_categorical("train.batch_size", [32, 64])

    # -----------------------------------------------------------------------
    # [anomaly_detection] - 保留的关键检测逻辑参数
    # -----------------------------------------------------------------------
    param_updates["anomaly_detection.anomaly_ratio"] = trial.suggest_float("ad.anomaly_ratio", 1.5, 3.5)
    param_updates["anomaly_detection.aggregation_method"] = trial.suggest_categorical(
        "ad.aggregation_method", ["mean", "max", "weighted_max"]
    )

    # 2. 创建并评估当前试验的配置
    current_config = update_config(base_config, param_updates)

    try:
        # swift_find_anomalies 封装了训练和预测
        predictions = swift_find_anomalies(data=data, config=current_config)
        score = affiliation_f(labels, predictions)
        return score
    except Exception as e:
        # 如果发生任何错误 (如 CUDA OOM), 打印错误并告诉 Optuna 该试验失败了
        print(f"Trial {trial.number} failed with an exception: {e}")
        # 返回一个很差的分数或直接剪枝
        raise optuna.TrialPruned()


# --------------------------------------------------------------------------- #
#                               主程序执行逻辑                                #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 Optuna 对 SWIFT 进行超参数优化", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-c", "--config", type=str, required=True, help="基础配置文件路径")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="【完整】数据集文件路径 (用于训练+验证)"
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="最佳配置的输出路径")
    parser.add_argument("--n-trials", type=int, default=100, help="要运行的总试验次数")
    parser.add_argument(
        "--study-name", type=str, default="swift-tuning-study", help="Optuna 研究的名称 (用于续传)"
    )
    parser.add_argument(
        "--no-save-intermediate", action="store_true", help="如果设置，则不在每次发现更优解时保存"
    )

    args = parser.parse_args()

    # 在开始时设置一次随机种子
    set_seed(1037)

    # 1. 加载数据和基础配置
    print("🚀 开始 SWIFT 参数自动调优 (使用 Optuna)...")
    base_config = load_config(args.config)
    df = pd.read_csv(args.dataset)
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].to_numpy()
    print(f"✅ 数据集 {args.dataset} 加载完成, 形状: {data.shape}")
    print(f"✅ 基础配置 {args.config} 加载完成")

    # 2. 设置并创建 Optuna Study
    # 使用 SQLite 数据库来存储研究结果，这样可以随时中断和恢复
    storage_name = f"sqlite:///{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_name,
        direction="maximize",  # 我们的目标是最大化 F1 分数
        load_if_exists=True,  # 如果研究已存在，则加载它，实现断点续传
        sampler=optuna.samplers.TPESampler(),  # 使用 TPE (一种贝叶斯优化算法)
        pruner=optuna.pruners.MedianPruner(),  # 使用中位数剪枝器 (可提前终止无望的试验)
    )
    print(f"📈 Optuna Study '{args.study_name}' 已设置。")
    print(f"   - 数据库: {storage_name}")
    print(f"   - 已完成试验数: {len(study.trials)}")
    print(f"   - 本次将运行试验数: {args.n_trials}")

    # 3. 定义回调函数，用于实时保存最佳配置
    def save_best_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """每当发现更好的分数时，保存一次最优配置。"""
        # 检查当前试验是否是迄今为止最好的
        if study.best_trial.number == trial.number:
            print(f"🎉 Trial {trial.number} 发现新的最佳分数: {trial.value:.4f}")
            save_optimal_config(base_config, trial.params, args.output)

    # 4. 开始优化
    print("\n🔥 开始优化... 按 Ctrl+C 可随时安全退出并续传。")

    # 将额外参数通过 lambda 传入 objective 函数
    objective_func = lambda trial: objective(trial, base_config, data, labels)

    callbacks = []
    if not args.no_save_intermediate:
        callbacks.append(save_best_callback)

    study.optimize(objective_func, n_trials=args.n_trials, callbacks=callbacks)

    # 5. 打印最终结果
    print("\n🎉 搜索完成!")
    print(f"   - 总试验次数: {len(study.trials)}")

    # 检查是否找到了任何有效的试验
    if study.best_trial:
        print(f"🏆 最佳 F1 分数: {study.best_value:.4f}")
        print("📋 最佳参数组合:")
        for key, value in study.best_params.items():
            print(f"   - {key}: {value}")

        # 确保最终的最佳配置被保存
        final_best_params_renamed = {
            k.replace("ad.", "anomaly_detection.")
            .replace("cfm.", "model.CFM.")
            .replace("train.", "training."): v
            for k, v in study.best_params.items()
        }
        save_optimal_config(base_config, final_best_params_renamed, args.output)
    else:
        print("❌ 未能完成任何有效的试验，无法确定最佳参数。")
