import argparse
import copy
import gc
import os
from typing import Any, Dict, List, Union

import numpy as np
import optuna
import pandas as pd
import tomli
import tomli_w
import torch
from baselines.swift.swift_pipeline import swift_find_anomalies, swift_score_anomalies
from evaluation.metrics.anomaly_detection_metrics_label import affiliation_f
from evaluation.metrics.anomaly_detection_metrics_score import auc_roc
from tools.tools import set_seed


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# ========== 统一参数配置 ==========
PARAM_CONFIG = {
    "seq_len": {
        "type": "categorical",
        "choices": [64, 128],
        "config_path": "data.seq_len",
    },
    "patch_size": {
        "type": "categorical",
        "choices": [8, 16, 32],
        "config_path": "data.patch_size",
    },
    "patch_stride": {
        "type": "categorical",
        "choices": [4, 8, 16],
        "config_path": "data.patch_stride",
    },
    "d_cf": {
        "type": "categorical",
        "choices": [32, 64, 96],
        "config_path": "model.CFM.d_cf",
    },
    "d_model": {
        "type": "categorical",
        "choices": [64, 96, 128],
        "config_path": "model.CFM.d_model",
    },
    "cfm_num_heads": {
        "type": "categorical",
        "choices": [2, 4],
        "config_path": "model.CFM.num_heads",
    },
    "cfm_attention_dropout": {
        "type": "float",
        "low": 0.1,
        "high": 0.2,
        "config_path": "model.CFM.attention_dropout",
        "decimal_places": 3,
    },
    "cfm_num_layers": {
        "type": "int",
        "low": 3,
        "high": 5,
        "config_path": "model.CFM.num_layers",
    },
    "cfm_dropout": {
        "type": "float",
        "low": 0.1,
        "high": 0.2,
        "config_path": "model.CFM.dropout",
        "decimal_places": 3,
    },
    "cfm_num_gat_heads": {
        "type": "categorical",
        "choices": [2, 4],
        "config_path": "model.CFM.num_gat_heads",
    },
    "cfm_gat_head_dim": {
        "type": "categorical",
        "choices": [8, 16],
        "config_path": "model.CFM.gat_head_dim",
    },
    "fm_level": {
        "type": "int",
        "low": 2,
        "high": 4,
        "config_path": "model.FM.level",
    },
    "fm_wavelet": {
        "type": "categorical",
        "choices": ["db4", "sym4", "coif2"],
        "config_path": "model.FM.wavelet",
    },
    # 时序重构参数
    "tsrm_is_flatten_individual": {
        "type": "categorical",
        "choices": [True, False],
        "config_path": "model.TSRM.is_flatten_individual",
    },
    # 损失函数参数
    "ccd_loss_lambda": {
        "type": "float",
        "low": 1e-4,
        "high": 1e-1,
        "log": True,
        "config_path": "loss.ccd_loss_lambda",
        "decimal_places": 5,
    },
    "scale_loss_lambda": {
        "type": "float",
        "low": 0.2,
        "high": 1.0,
        "config_path": "loss.scale_loss_lambda",
        "decimal_places": 3,
    },
    "ccd_regular_lambda": {
        "type": "float",
        "low": 0.05,
        "high": 0.5,
        "config_path": "loss.ccd_regular_lambda",
        "decimal_places": 3,
    },
    "ccd_align_lambda": {
        "type": "float",
        "low": 0.5,
        "high": 2.0,
        "config_path": "loss.ccd_align_lambda",
        "decimal_places": 3,
    },
    # 训练参数
    "learning_rate": {
        "type": "float",
        "low": 1e-4,
        "high": 5e-3,
        "log": True,
        "config_path": "training.learning_rate",
        "decimal_places": 5,
    },
    # 异常检测参数
    "scale_score_lambda": {
        "type": "float",
        "low": 0.1,
        "high": 1.0,
        "config_path": "anomaly_detection.scale_score_lambda",
        "decimal_places": 3,
    },
}


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 toml 配置文件"""
    try:
        with open(config_path, "rb") as f:
            return tomli.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    except Exception as e:
        raise ValueError(f"配置文件格式错误: {e}")


def update_config(base_config: Dict[str, Any], param_updates: Dict[str, Any]) -> Dict[str, Any]:
    """更新 toml 配置文件"""
    config = copy.deepcopy(base_config)
    for key_path, value in param_updates.items():
        # 支持使用诸如 "model.CFM.d_cf" 这样的点语法更新嵌套字段
        keys = key_path.split(".")
        current = config
        # 依次向下查找/创建嵌套字典
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        # 更新最终键对应的值
        current[keys[-1]] = value
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """保存 toml 配置文件"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            tomli_w.dump(config, f)
        print(f"💾 配置已保存到: {output_path}")
    except Exception as e:
        print(f"❌ 保存配置失败: {e}")


def suggest_parameter(
    trial: optuna.trial.Trial, param_name: str, param_config: Dict[str, Any]
) -> Any:
    """根据参数配置动态建议参数值"""
    param_type = param_config["type"]

    if param_type == "categorical":
        return trial.suggest_categorical(param_name, param_config["choices"])
    elif param_type == "int":
        return trial.suggest_int(param_name, param_config["low"], param_config["high"])
    elif param_type == "float":
        log = param_config.get("log", False)
        value = trial.suggest_float(param_name, param_config["low"], param_config["high"], log=log)
        # 如果配置了小数位数，则进行四舍五入
        decimal_places = param_config.get("decimal_places")
        if decimal_places is not None:
            value = round(value, decimal_places)
        return value
    else:
        raise ValueError(f"不支持的参数类型: {param_type}")


def validate_params(params: Dict[str, Any]) -> bool:
    """验证参数组合的有效性"""
    seq_len = params["seq_len"]
    patch_size = params["patch_size"]
    patch_stride = params["patch_stride"]
    d_model = params["d_model"]
    d_cf = params["d_cf"]

    # 基本约束检查
    if patch_size >= seq_len:
        return False
    if (seq_len - patch_size) % patch_stride != 0:
        return False
    if patch_stride > patch_size:
        return False
    if d_model < d_cf:
        return False

    return True


def objective(
    trial: optuna.trial.Trial,
    base_config: Dict[str, Any],
    all_data: np.ndarray,
    labels: np.ndarray,
    task_name: str,
) -> float:
    """Optuna目标函数"""

    # 在每次试验开始前清理GPU内存
    clear_gpu_memory()

    # 动态生成所有参数
    params = {}
    for param_name, param_config in PARAM_CONFIG.items():
        params[param_name] = suggest_parameter(trial, param_name, param_config)

    # 验证参数组合
    if not validate_params(params):
        raise optuna.TrialPruned("无效的参数组合")

    # 构建参数更新字典
    param_updates = {}
    for param_name, value in params.items():
        config_path = PARAM_CONFIG[param_name]["config_path"]
        param_updates[config_path] = value

    # 更新配置
    current_config = update_config(base_config, param_updates)
    # 设置 trial_number 用于日志记录
    current_config["trial_number"] = trial.number

    try:
        if task_name == "find_anomalies":
            # 异常检测任务 - 使用 F1 分数
            predictions = swift_find_anomalies(all_data=all_data, config=current_config)
            result = affiliation_f(labels, predictions)
        elif task_name == "score_anomalies":
            # 异常评分任务 - 使用 AUC-ROC
            scores = swift_score_anomalies(all_data=all_data, config=current_config)
            result = auc_roc(labels, scores)
        else:
            raise ValueError(f"不支持的任务类型: {task_name}")

        clear_gpu_memory()  # 清理内存
        return float(result)

    except torch.cuda.OutOfMemoryError as e:
        print(f"💥 Trial {trial.number} GPU内存不足: {str(e)[:100]}...")
        clear_gpu_memory()
        raise optuna.TrialPruned(f"GPU内存不足: {e}")

    except Exception as e:
        print(f"❌ Trial {trial.number} 执行失败: {e}")
        clear_gpu_memory()
        raise optuna.TrialPruned(f"试验执行失败: {e}")


def run_optimization(args: argparse.Namespace):
    """运行 Optuna 超参数优化流程"""
    print("🚀 开始超参数调优...")
    print(f"🎯 任务: {args.task_name} | 数据集: {args.dataset_name} | 算法: {args.algorithm_name}")
    print(f"🔬 异常率: {args.anomaly_ratio}% | 试验次数: {args.n_trials}")

    # 验证任务类型
    if args.task_name not in ["find_anomalies", "score_anomalies"]:
        print(f"❌ 不支持的任务类型: {args.task_name}")
        print("📝 支持的任务类型: find_anomalies, score_anomalies")
        return

    # 根据任务类型确定评估指标名称
    metric_name = "Aff-F" if args.task_name == "find_anomalies" else "A-R"

    # ---------- 加载基础配置与数据 ----------
    try:
        base_config = load_config(args.base_config)
        # 异常率设置
        base_config["anomaly_detection"]["anomaly_ratio"] = args.anomaly_ratio
        # 训练验证集长度设置
        base_config["data"]["tain_val_len"] = args.train_val_len

        df = pd.read_csv(args.dataset_path)
        all_data = df.iloc[:, :-1].values  # 训练验证测试集
        test_labels = df.iloc[args.train_val_len :, -1].to_numpy()  # 测试集标签

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # ---------- 创建 Optuna Study ----------
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )

    # ---------- 回调用于追踪最佳结果 ----------
    best_value: float = -1.0
    best_params: Dict[str, Any] = {}
    best_config_path: str = ""  # 记录当前最佳配置文件路径

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        nonlocal best_value, best_params, best_config_path
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and trial.value > best_value
        ):
            # 删除之前的最佳配置文件
            if best_config_path and os.path.exists(best_config_path):
                try:
                    os.remove(best_config_path)
                    print(f"🗑️ 删除之前的配置: {os.path.basename(best_config_path)}")
                except Exception as e:
                    print(f"⚠️ 删除文件失败: {e}")

            best_value = trial.value
            best_params = trial.params.copy()
            print(f"🎉 发现新的最佳结果: {metric_name} = {best_value:.3f} (Trial {trial.number})")

            param_updates = {}
            for param_name, value in trial.params.items():
                if param_name in PARAM_CONFIG:
                    config_path = PARAM_CONFIG[param_name]["config_path"]
                    param_updates[config_path] = value

            current_best_config = update_config(base_config, param_updates)

            # 生成包含指标分数的文件名
            base_output_config = args.output_config.replace(".toml", "")
            metric_suffix = "aff-f" if args.task_name == "find_anomalies" else "a-r"
            output_config_with_score = f"{base_output_config}_{metric_suffix}_{best_value:.3f}.toml"
            best_config_path = output_config_with_score  # 更新最佳配置路径

            save_config(current_best_config, output_config_with_score)

    # ---------- 执行优化 ----------
    try:
        study.optimize(
            lambda t: objective(t, base_config, all_data, test_labels, args.task_name),
            n_trials=args.n_trials,
            callbacks=[_callback],
        )
    except KeyboardInterrupt:
        print("⏹️ 优化被用户中断")
    except Exception as e:
        print(f"❌ 优化过程出错: {e}")

    # ---------- 最终结果总结 ----------
    if best_params:
        print("\n✅ 调优完成!")
        print(f"📊 最佳 {metric_name} 分数: {best_value:.3f}")
    else:
        print("\n⚠️ 没有找到有效的最佳参数，请检查数据和配置")


# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="SWIFT 超参数调优工具")

    # 基本参数
    cli_parser.add_argument("--task-name", type=str, required=True, help="任务名称")
    cli_parser.add_argument("--dataset-name", type=str, required=True, help="数据集名称")
    cli_parser.add_argument("--train-val-len", type=int, required=True, help="训练验证集总长度")
    cli_parser.add_argument("--algorithm-name", type=str, required=True, help="算法名称")
    cli_parser.add_argument("--anomaly-ratio", type=float, required=True, help="异常率 (百分比)")
    cli_parser.add_argument("--n-trials", type=int, required=True, help="Optuna 试验次数")

    # 路径参数
    cli_parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base/{algorithm_name}.toml",
        help="基础配置文件路径",
    )
    cli_parser.add_argument(
        "--output-config",
        type=str,
        default="configs/{task_name}/{dataset_name}/{algorithm_name}.toml",
        help="输出配置文件路径",
    )
    cli_parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/{dataset_name}.csv",
        help="数据集文件路径",
    )

    _args = cli_parser.parse_args()

    set_seed(1037)  # 随机种子

    _args.base_config = _args.base_config.format(
        task_name=_args.task_name,
        dataset_name=_args.dataset_name,
        algorithm_name=_args.algorithm_name,
    )
    # 先格式化基本信息
    _args.output_config = _args.output_config.format(
        task_name=_args.task_name,
        dataset_name=_args.dataset_name,
        algorithm_name=_args.algorithm_name,
    )

    # 根据任务类型决定是否添加 ratio 字段
    if _args.task_name == "find_anomalies":
        # 为 find_anomalies 任务添加 ratio 字段
        base_name = _args.output_config.replace(".toml", "")
        _args.output_config = f"{base_name}_ratio_{_args.anomaly_ratio}.toml"
    _args.dataset_path = _args.dataset_path.format(dataset_name=_args.dataset_name)

    run_optimization(_args)
