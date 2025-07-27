import argparse
import copy
import gc
import os
from typing import Any, Dict, List, Union

import numpy as np
import optuna
import optuna.visualization as vis
import pandas as pd
import plotly.graph_objects as go
import tomli
import tomli_w
import torch
from baselines.swift.swift_pipeline import swift_find_anomalies, swift_score_anomalies
from evaluation.metrics.anomaly_detection_metrics_label import affiliation_f
from evaluation.metrics.anomaly_detection_metrics_score import auc_roc
from tools.tools import set_seed

# TODO: 遇到 cuda out of memory, 那么此时的参数组合是否需要重试?


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def create_study_name(args: argparse.Namespace) -> str:
    """创建study名称"""
    return f"{args.task_name}_{args.dataset_name}_{args.algorithm_name}_ratio_{args.anomaly_ratio}"


def create_study_db_path(args: argparse.Namespace) -> str:
    """创建study数据库路径"""
    study_name = create_study_name(args)
    db_dir = "optuna_studies"
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, f"{study_name}.db")


def generate_optimization_plots(study: optuna.Study, args: argparse.Namespace) -> None:
    """生成优化过程的可视化图表"""
    if len(study.trials) == 0:
        print("⚠️ 没有试验数据，跳过可视化图表生成")
        return

    # 创建可视化输出目录
    viz_dir = f"optuna_visualizations/{args.dataset_name}/{args.algorithm_name}"
    os.makedirs(viz_dir, exist_ok=True)

    # 根据任务类型确定指标名称
    metric_name = "Affiliation-F" if args.task_name == "find_anomalies" else "AUC-ROC"

    try:
        # 1. 优化历史图 - 显示每次试验的结果和最佳值趋势
        print("📊 生成优化历史图...")
        fig_history = vis.plot_optimization_history(study)
        fig_history.update_layout(
            title=f"优化历史 - {args.dataset_name} ({metric_name})",
            xaxis_title="试验次数",
            yaxis_title=f"{metric_name} 分数",
        )
        fig_history.write_html(f"{viz_dir}/optimization_history.html")

        # 2. 参数重要性图 - 显示哪些参数对结果影响最大
        print("📊 生成参数重要性图...")
        fig_importance = vis.plot_param_importances(study)
        fig_importance.update_layout(
            title=f"参数重要性 - {args.dataset_name}", xaxis_title="重要性"
        )
        fig_importance.write_html(f"{viz_dir}/param_importances.html")

        # 3. 参数关系图 - 显示参数之间的相关性
        print("📊 生成参数关系图...")
        fig_slice = vis.plot_slice(study)
        fig_slice.update_layout(title=f"参数切片分析 - {args.dataset_name}")
        fig_slice.write_html(f"{viz_dir}/param_slice.html")

        # 4. 并行坐标图 - 显示高性能试验的参数组合
        print("📊 生成并行坐标图...")
        fig_parallel = vis.plot_parallel_coordinate(study)
        fig_parallel.update_layout(title=f"并行坐标图 - {args.dataset_name}")
        fig_parallel.write_html(f"{viz_dir}/parallel_coordinate.html")

        # 5. 收敛分析 - 自定义图表分析收敛情况
        print("📊 生成收敛分析图...")
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) > 1:
            trial_numbers = [t.number for t in completed_trials]
            trial_values = [t.value for t in completed_trials]

            # 计算运行最佳值
            best_values = []
            current_best = float("-inf")
            for value in trial_values:
                if value is not None and value > current_best:
                    current_best = value
                best_values.append(current_best)

            # 创建收敛图
            fig_convergence = go.Figure()
            fig_convergence.add_trace(
                go.Scatter(
                    x=trial_numbers,
                    y=trial_values,
                    mode="markers",
                    name="试验结果",
                    marker=dict(color="lightblue", size=8),
                )
            )
            fig_convergence.add_trace(
                go.Scatter(
                    x=trial_numbers,
                    y=best_values,
                    mode="lines+markers",
                    name="最佳值趋势",
                    line=dict(color="red", width=2),
                )
            )
            fig_convergence.update_layout(
                title=f"收敛分析 - {args.dataset_name}",
                xaxis_title="试验次数",
                yaxis_title=f"{metric_name} 分数",
                hovermode="x unified",
            )
            fig_convergence.write_html(f"{viz_dir}/convergence_analysis.html")

        print(f"📊 可视化图表已保存到: {viz_dir}")
        print(f"📊 可在浏览器中打开 HTML 文件查看交互式图表")

    except Exception as e:
        print(f"❌ 生成可视化图表时出错: {e}")


def analyze_convergence(study: optuna.Study, window_size: int = 20) -> Dict[str, Any]:
    """分析优化收敛情况"""
    completed_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]

    if len(completed_trials) < window_size:
        return {
            "is_converged": False,
            "reason": f"试验数量不足 ({len(completed_trials)} < {window_size})",
            "improvement_rate": 0.0,
        }

    # 计算最近window_size个试验的改进情况
    recent_trials = completed_trials[-window_size:]
    recent_values = [t.value for t in recent_trials if t.value is not None]

    if not recent_values:
        return {
            "is_converged": False,
            "reason": "没有有效的试验值",
            "improvement_rate": 0.0,
        }

    # 计算改进率
    best_in_window = max(recent_values)
    all_values = [t.value for t in completed_trials if t.value is not None]
    best_overall = max(all_values) if all_values else 0.0
    improvement_rate = (
        (best_in_window - best_overall) / abs(best_overall) if best_overall != 0 else 0
    )

    # 判断是否收敛
    is_converged = abs(improvement_rate) < 0.001  # 改进率小于0.1%认为收敛

    return {
        "is_converged": is_converged,
        "improvement_rate": improvement_rate,
        "recent_best": best_in_window,
        "overall_best": best_overall,
        "trials_analyzed": len(completed_trials),
    }


def get_n_trials_recommendation(dataset_name: str, algorithm_name: str) -> int:
    """根据数据集和算法推荐试验次数"""
    # 基础试验次数
    base_trials = 50

    # 根据数据集大小调整
    dataset_multipliers = {
        "PSM": 2.0,  # 大数据集
        "MSL": 1.5,  # 中等数据集
        "CalIt2": 1.0,  # 小数据集
    }

    # 根据算法复杂度调整
    algorithm_multipliers = {
        "swift": 1.5,  # 复杂算法，需要更多试验
    }

    multiplier = dataset_multipliers.get(dataset_name, 1.0) * algorithm_multipliers.get(
        algorithm_name, 1.0
    )
    recommended_trials = int(base_trials * multiplier)

    return recommended_trials


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
        "choices": [64, 96, 128, 256],
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
        "high": 0.4,
        "config_path": "model.CFM.attention_dropout",
        "decimal_places": 3,
    },
    "cfm_num_layers": {
        "type": "int",
        "low": 2,
        "high": 8,
        "config_path": "model.CFM.num_layers",
    },
    "cfm_dropout": {
        "type": "float",
        "low": 0.1,
        "high": 0.4,
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
        "low": 5e-5,
        "high": 1e-2,
        "log": True,
        "config_path": "training.learning_rate",
        "decimal_places": 5,
    },
    "weight_decay": {
        "type": "float",
        "low": 0.0,
        "high": 5e-2,
        "log": False,
        "config_path": "training.weight_decay",
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
    "score_aggregation_alpha": {
        "type": "float",
        "low": 0.1,
        "high": 0.9,
        "config_path": "anomaly_detection.score_aggregation_alpha",
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
        # 打印数据集的形状
        print(f"数据集变量数: {all_data.shape[1]}")
        test_labels = df.iloc[args.train_val_len :, -1].to_numpy()  # 测试集标签

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # ---------- 创建 Optuna Study ----------
    study_db_path = create_study_db_path(args)
    study_name = create_study_name(args)
    print(f"💾 Study DB 路径: {study_db_path}")

    # 检查是否强制重启
    if args.restart and os.path.exists(study_db_path):
        print("🔄 强制重新开始调优，删除现有数据库...")
        try:
            os.remove(study_db_path)
            print(f"🗑️ 删除文件: {study_db_path}")
        except Exception as e:
            print(f"❌ 删除文件失败: {e}")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            storage=f"sqlite:///{study_db_path}",
            study_name=study_name,
            load_if_exists=False,  # 强制不加载已存在的试验
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            storage=f"sqlite:///{study_db_path}",
            study_name=study_name,
            load_if_exists=True,
        )

    # 显示已完成的试验信息
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed_trials > 0:
        print(f"📊 从数据库恢复了 {completed_trials} 个已完成的试验")
        try:
            best_trial = study.best_trial
            if best_trial:
                print(f"🏆 当前最佳分数: {best_trial.value:.3f} (Trial {best_trial.number})")
        except ValueError:
            # 当没有完成的试验时，best_trial 会抛出异常
            print("⚠️ 尚未有完成的试验")
    else:
        print("🆕 开始新的调优会话")

    # 计算剩余试验数
    remaining_trials = max(0, args.n_trials - len(study.trials))
    if remaining_trials == 0:
        print("✅ 所有试验已完成!")
        try:
            if study.best_trial:
                print(f"🏆 最佳分数: {study.best_trial.value:.3f}")
        except ValueError:
            print("⚠️ 没有完成的试验")
        return
    elif remaining_trials < args.n_trials:
        print(f"🔄 将继续执行剩余的 {remaining_trials} 个试验")

    # ---------- 回调用于追踪最佳结果 ----------
    # 从数据库获取当前最佳值，安全地处理没有试验的情况
    best_value: float = -1.0
    best_params: Dict[str, Any] = {}
    best_config_path: str = ""  # 记录当前最佳配置文件路径
    try:
        if study.best_trial:
            best_value = study.best_value
            best_params = study.best_params
    except ValueError:
        # 没有完成的试验时会抛出异常
        pass

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
                    # 应用精度处理
                    param_config = PARAM_CONFIG[param_name]
                    if param_config["type"] == "float":
                        decimal_places = param_config.get("decimal_places")
                        if decimal_places is not None:
                            value = round(value, decimal_places)
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
            n_trials=remaining_trials,
            callbacks=[_callback],
        )
    except KeyboardInterrupt:
        print("⏹️ 优化被用户中断")
        print(f"💾 进度已保存到数据库: {study_db_path}")
    except Exception as e:
        print(f"❌ 优化过程出错: {e}")
        print(f"💾 进度已保存到数据库: {study_db_path}")

    # ---------- 生成可视化图表 ----------
    if args.enable_visualization:
        print("\n📊 生成可视化图表...")
        generate_optimization_plots(study, args)
    else:
        print("\n📊 跳过可视化图表生成（使用 --enable-visualization 启用）")

    # ---------- 收敛分析 ----------
    convergence_info = analyze_convergence(study)
    print(f"\n🔍 收敛分析:")
    print(f"  - 是否收敛: {'是' if convergence_info['is_converged'] else '否'}")
    if not convergence_info["is_converged"]:
        print(f"  - 原因: {convergence_info.get('reason', '改进率过高')}")
    print(f"  - 改进率: {convergence_info['improvement_rate']:.4f}")
    print(f"  - 分析试验数: {convergence_info['trials_analyzed']}")

    # ---------- 试验次数建议 ----------
    recommended_trials = get_n_trials_recommendation(args.dataset_name, args.algorithm_name)
    current_trials = len(study.trials)
    if current_trials < recommended_trials:
        print(f"\n💡 建议: 当前试验数 ({current_trials}) 少于推荐数 ({recommended_trials})")
        print(f"   考虑增加试验次数以获得更好的结果")

    # ---------- 最终结果总结 ----------
    try:
        final_best_trial = study.best_trial
        if final_best_trial:
            print("\n✅ 调优完成!")
            print(f"📊 最佳 {metric_name} 分数: {final_best_trial.value:.3f}")
            print(f"📈 总共完成了 {len(study.trials)} 个试验")
            print(f"💾 调优进度已保存到: {study_db_path}")
        else:
            print("\n⚠️ 没有找到有效的最佳参数，请检查数据和配置")
    except ValueError:
        # 没有完成的试验时会抛出异常
        print("\n⚠️ 没有找到有效的最佳参数，请检查数据和配置")
        if len(study.trials) > 0:
            print(f"📈 总共尝试了 {len(study.trials)} 个试验")
            print(f"💾 调优进度已保存到: {study_db_path}")


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

    # 调优控制参数
    cli_parser.add_argument(
        "--restart", action="store_true", help="强制重新开始调优（忽略已保存的进度）"
    )
    cli_parser.add_argument(
        "--enable-visualization", action="store_true", help="启用可视化图表生成（需要安装plotly）"
    )

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
