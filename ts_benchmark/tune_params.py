import argparse
import copy
import os
from typing import Any, Dict, List

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


def objective(
    trial: optuna.trial.Trial, base_config: Dict[str, Any], data: np.ndarray, labels: np.ndarray
) -> float:
    """Optuna 核心目标函数"""
    param_updates = {}
    param_updates["trial_number"] = trial.number

    # ---- data 数据处理 ----
    seq_len = trial.suggest_categorical("data.seq_len", [64, 128, 256])

    possible_patch_sizes = [8, 16, 32]
    valid_patch_sizes = [p for p in possible_patch_sizes if p < seq_len]
    if not valid_patch_sizes:
        raise optuna.TrialPruned("No valid patch_size for the chosen seq_len.")
    patch_size = trial.suggest_categorical("data.patch_size", valid_patch_sizes)

    divisible_length = seq_len - patch_size

    # 寻找所有有效的 patch_stride (即 divisible_length 的所有约数)
    valid_strides = []
    for i in range(1, int(divisible_length**0.5) + 1):
        if divisible_length % i == 0:
            valid_strides.append(i)
            if i * i != divisible_length:
                valid_strides.append(divisible_length // i)
    valid_strides.sort()

    practical_strides = [s for s in valid_strides if 4 <= s <= patch_size]
    if not practical_strides:
        if not valid_strides:
            raise optuna.TrialPruned("No valid strides found.")
        patch_stride = trial.suggest_categorical("data.patch_stride", valid_strides)
    else:
        patch_stride = trial.suggest_categorical("data.patch_stride", practical_strides)

    param_updates["data.seq_len"] = seq_len
    param_updates["data.patch_size"] = patch_size
    param_updates["data.patch_stride"] = patch_stride

    # ---- loss 损失函数 ----
    param_updates["loss.ccd_loss_lambda"] = trial.suggest_float("loss.ccd_loss_lambda", 1e-4, 1e-1, log=True)
    param_updates["loss.scale_loss_lambda"] = trial.suggest_float("loss.scale_loss_lambda", 0.2, 1.0)
    param_updates["loss.ccd_regular_lambda"] = trial.suggest_float("loss.ccd_regular_lambda", 0.05, 0.5)
    param_updates["loss.ccd_align_lambda"] = trial.suggest_float("loss.ccd_align_lambda", 0.5, 2.0)

    # ---- model.FM 前向模块 ----
    param_updates["model.FM.level"] = trial.suggest_int("fm.level", 2, 5)
    param_updates["model.FM.wavelet"] = trial.suggest_categorical("fm.wavelet", ["db4", "sym4", "coif2"])

    # ---- model.CFM 通道融合模块 ----
    param_updates["model.CFM.num_heads"] = trial.suggest_categorical("cfm.num_heads", [2, 4, 8])
    param_updates["model.CFM.attention_dropout"] = trial.suggest_float("cfm.attention_dropout", 0.05, 0.3)
    d_cf_val = trial.suggest_categorical("cfm.d_cf", [64, 96, 128])
    d_model_val = trial.suggest_categorical("cfm.d_model", [64, 96, 128])
    if d_model_val < d_cf_val:
        raise optuna.TrialPruned()
    param_updates["model.CFM.d_cf"] = d_cf_val
    param_updates["model.CFM.d_model"] = d_model_val
    param_updates["model.CFM.num_layers"] = trial.suggest_int("cfm.num_layers", 3, 6)
    param_updates["model.CFM.dropout"] = trial.suggest_float("cfm.dropout", 0.1, 0.25)
    param_updates["model.CFM.num_gat_heads"] = trial.suggest_categorical("cfm.num_gat_heads", [2, 4, 8])
    param_updates["model.CFM.gat_head_dim"] = trial.suggest_categorical("cfm.gat_head_dim", [8, 16, 32])

    # ---- model.TSRM 时尺重构模块 ----
    param_updates["model.TSRM.is_flatten_individual"] = trial.suggest_categorical(
        "tsrm.is_flatten_individual", [True, False]
    )

    # ---- training 训练配置 ----
    param_updates["training.learning_rate"] = trial.suggest_float(
        "training.learning_rate", 1e-4, 5e-3, log=True
    )
    param_updates["training.batch_size"] = trial.suggest_categorical("training.batch_size", [32, 64])

    # ---- anomaly_detection 异常检测配置 ----
    param_updates["anomaly_detection.scale_score_lambda"] = trial.suggest_float(
        "anomaly_detection.scale_score_lambda", 0.1, 1.0
    )

    # 创建并评估当前试验的配置
    current_config = update_config(base_config, param_updates)

    try:
        predictions = swift_find_anomalies(data=data, config=current_config)
        aff_f1 = affiliation_f(labels, predictions)
        return aff_f1
    except Exception as e:
        print(f"Trial {trial.number} failed with an exception: {e}")
        raise optuna.TrialPruned()


def run_tuning_for_ratio(
    ratio: float, base_config: Dict[str, Any], data: np.ndarray, labels: np.ndarray, args: argparse.Namespace
) -> None:
    """为指定的异常率运行超参数调优"""
    print(f"🚀 ========================================================")
    print(f"🚀 开始为 Anomaly Ratio = {ratio}% 进行超参数调优...")
    print(f"🚀 ========================================================")

    # 格式化当前异常率的路径
    current_output_config = args.output_config.format(ratio=ratio)
    current_study_name = args.study_name.format(ratio=ratio)

    # 创建当前异常率的配置
    current_config = copy.deepcopy(base_config)
    current_config["anomaly_detection"]["anomaly_ratio"] = ratio
    current_config["anomaly_detection"]["threshold_strategy"] = args.threshold_strategy
    current_config["anomaly_detection"]["aggregation_method"] = args.aggregation_method

    # 创建 Optuna 研究
    storage_name = f"sqlite:///{current_study_name}.db"
    study = optuna.create_study(
        study_name=current_study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )

    # 定义回调函数，用于实时保存最佳配置
    def save_best_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if study.best_trial.number == trial.number:
            print(f"🎉 Trial {trial.number} 发现新的最佳 Affiliation F1 分数: {trial.value:.4f}")

            # 重映射参数键
            key_map = {
                "cfm.": "model.CFM.",
                "fm.": "model.FM.",
                "tsrm.": "model.TSRM.",
            }

            mapped_params = {}
            for k, v in trial.params.items():
                new_key = k
                for short, full in key_map.items():
                    if k.startswith(short):
                        new_key = k.replace(short, full)
                        break
                mapped_params[new_key] = v

            save_optimal_config(current_config, mapped_params, current_output_config)

    # 创建目标函数
    objective_func = lambda trial: objective(trial, current_config, data, labels)

    # 运行优化
    study.optimize(objective_func, n_trials=args.n_trials, callbacks=[save_best_callback])

    print(f"✅ Anomaly Ratio = {ratio}% 的调优完成。最佳配置已保存到 {current_output_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 Optuna 对 SWIFT 进行批量超参数优化", formatter_class=argparse.RawTextHelpFormatter
    )

    # 必需参数
    parser.add_argument("--task-name", type=str, required=True, help="任务名称 (如: find_anomalies)")
    parser.add_argument("--dataset-name", type=str, required=True, help="数据集名称 (如: ASD_dataset_1)")
    parser.add_argument("--algorithm-name", type=str, required=True, help="异常检测算法名称 (如: swift)")

    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/{task_name}/{dataset_name}/{algorithm_name}_base.toml",
        help="基础配置文件路径",
    )
    # 输出配置
    parser.add_argument(
        "--output-config",
        type=str,
        default="configs/{task_name}/{dataset_name}/{algorithm_name}_best_ar_{ratio}.toml",
        help="输出配置文件路径",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/{dataset_name}.csv",
        help="数据集文件路径 (用于训练+验证)",
    )

    # 异常率配置
    parser.add_argument(
        "--anomaly-ratios", type=str, default="1,3,5,8", help="要调优的异常率列表，用逗号分隔 (默认: 1,3,5,8)"
    )

    # 固定策略参数
    parser.add_argument(
        "--threshold-strategy", type=str, default="adaptive", help="阈值计算策略 (默认: adaptive)"
    )
    parser.add_argument(
        "--aggregation-method", type=str, default="weighted_max", help="分数聚合方法 (默认: weighted_max)"
    )

    # 调优配置
    parser.add_argument("--n-trials", type=int, default=100, help="每个异常率的试验次数 (默认: 100)")

    parser.add_argument(
        "--study-name",
        type=str,
        default="{task_name}-{dataset_name}-{algorithm_name}-ar{ratio}",
        help="Optuna 研究名称",
    )

    # 可选参数
    parser.add_argument("--seed", type=int, default=1037, help="随机种子 (默认: 1037)")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 格式化路径模板
    args.base_config = args.base_config.format(
        task_name=args.task_name, dataset_name=args.dataset_name, algorithm_name=args.algorithm_name
    )

    args.output_config = args.output_config.format(
        task_name=args.task_name,
        dataset_name=args.dataset_name,
        algorithm_name=args.algorithm_name,
        ratio="{ratio}",  # 保留 ratio 占位符，稍后在 run_tuning_for_ratio 中格式化
    )

    args.dataset_path = args.dataset_path.format(dataset_name=args.dataset_name)

    args.study_name = args.study_name.format(
        task_name=args.task_name,
        dataset_name=args.dataset_name,
        algorithm_name=args.algorithm_name,
        ratio="{ratio}",  # 保留 ratio 占位符，稍后在 run_tuning_for_ratio 中格式化
    )

    # 解析异常率列表
    try:
        anomaly_ratios = [float(x.strip()) for x in args.anomaly_ratios.split(",")]
    except ValueError:
        print("❌ 异常率格式错误，请使用逗号分隔的数字列表")

    print(f"🚀 参数自动调优开始...")
    print(f"🎯 任务: {args.task_name} | 数据集: {args.dataset_name} | 算法: {args.algorithm_name.upper()}")
    print(f"🔬 异常率: {anomaly_ratios}")
    print(f"🔬 每个异常率试验次数: {args.n_trials}")
    print(f"🎛️ 阈值策略: {args.threshold_strategy}")
    print(f"🔗 聚合方法: {args.aggregation_method}")

    # 加载基础配置和数据
    base_config = load_config(args.base_config)
    df = pd.read_csv(args.dataset_path)
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].to_numpy()

    print(f"📁 数据形状: {data.shape}, 标签形状: {labels.shape}")

    # 依次为每个异常率运行调优
    for i, ratio in enumerate(anomaly_ratios, 1):
        print(f"\n[{i}/{len(anomaly_ratios)}] 处理异常率: {ratio}%")
        try:
            run_tuning_for_ratio(ratio, base_config, data, labels, args)
        except Exception as e:
            print(f"❌ 异常率 {ratio}% 调优失败: {e}")
            continue

    print(f"\n🎉🎉🎉 所有 {len(anomaly_ratios)} 个异常率的调优任务均已执行完毕! 🎉🎉🎉")
