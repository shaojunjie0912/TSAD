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
    param_updates = {}
    param_updates["trial_number"] = trial.number

    # ---- loss 损失函数 ----
    param_updates["loss.ccd_loss_lambda"] = trial.suggest_float("loss.ccd_loss_lambda", 1e-4, 1e-1, log=True)
    param_updates["loss.scale_loss_lambda"] = trial.suggest_float("loss.scale_loss_lambda", 0.2, 1.0)
    param_updates["loss.ccd_regular_lambda"] = trial.suggest_float("loss.ccd_regular_lambda", 0.05, 0.5)
    param_updates["loss.ccd_align_lambda"] = trial.suggest_float("loss.ccd_align_lambda", 0.5, 2.0)

    # ---- model.FM 前向模块 ----
    param_updates["model.FM.level"] = trial.suggest_int("fm.level", 2, 5)

    # ---- model.CFM 通道融合模块 ----
    param_updates["model.CFM.num_heads"] = trial.suggest_categorical("cfm.num_heads", [2, 4, 8])
    param_updates["model.CFM.attention_dropout"] = trial.suggest_float("cfm.attention_dropout", 0.05, 0.3)
    d_cf_val = trial.suggest_categorical("cfm.d_cf", [64, 96, 128])
    d_model_val = trial.suggest_categorical("cfm.d_model", [64, 96, 128])
    if d_model_val < d_cf_val:  # 不满足约束条件 d_model >= d_cf 则剪枝
        raise optuna.TrialPruned()
    param_updates["model.CFM.d_cf"] = d_cf_val
    param_updates["model.CFM.d_model"] = d_model_val
    param_updates["model.CFM.num_layers"] = trial.suggest_int("cfm.num_layers", 3, 6)
    param_updates["model.CFM.dropout"] = trial.suggest_float("cfm.dropout", 0.1, 0.25)

    # ---- training 训练配置 ----
    param_updates["training.learning_rate"] = trial.suggest_float(
        "training.learning_rate", 1e-4, 5e-3, log=True
    )
    param_updates["training.batch_size"] = trial.suggest_categorical("training.batch_size", [32, 64])

    # ---- anomaly_detection 异常检测配置 ----
    param_updates["anomaly_detection.anomaly_ratio"] = trial.suggest_float(
        "anomaly_detection.anomaly_ratio", 1.5, 3.5
    )
    param_updates["anomaly_detection.threshold_strategy"] = trial.suggest_categorical(
        "anomaly_detection.threshold_strategy", ["percentile", "robust_percentile", "std", "adaptive"]
    )
    param_updates["anomaly_detection.aggregation_method"] = trial.suggest_categorical(
        "anomaly_detection.aggregation_method", ["mean", "max", "weighted_max"]
    )

    # 2. 创建并评估当前试验的配置
    current_config = update_config(base_config, param_updates)

    try:
        predictions = swift_find_anomalies(data=data, config=current_config)
        score = affiliation_f(labels, predictions)
        return score
    except Exception as e:
        print(f"Trial {trial.number} failed with an exception: {e}")
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

    # 随机种子
    set_seed(1037)

    print("🚀 开始 SWIFT 参数自动调优...")
    base_config = load_config(args.config)
    df = pd.read_csv(args.dataset)
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].to_numpy()

    storage_name = f"sqlite:///{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_name,
        direction="maximize",  # 我们的目标是最大化 F1 分数
        load_if_exists=True,  # 如果研究已存在，则加载它，实现断点续传
        sampler=optuna.samplers.TPESampler(),  # 使用 TPE (一种贝叶斯优化算法)
        pruner=optuna.pruners.MedianPruner(),  # 使用中位数剪枝器 (可提前终止无望的试验)
    )

    # 3. 定义回调函数，用于实时保存最佳配置
    def save_best_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """每当发现更好的分数时，保存一次最优配置。"""
        # 检查当前试验是否是迄今为止最好的
        if study.best_trial.number == trial.number:
            print(f"🎉 Trial {trial.number} 发现新的最佳 Affiliation F1 分数: {trial.value:.4f}")
            # 参数名重映射
            mapped_params = {
                k.replace("cfm.", "model.CFM.").replace("fm.", "model.FM."): v
                for k, v in trial.params.items()
            }
            save_optimal_config(base_config, mapped_params, args.output)

    # 将额外参数通过 lambda 传入 objective 函数
    objective_func = lambda trial: objective(trial, base_config, data, labels)

    callbacks = []
    if not args.no_save_intermediate:
        callbacks.append(save_best_callback)

    study.optimize(objective_func, n_trials=args.n_trials, callbacks=callbacks)

    print("\n🎉 搜索完成!")
