#!/usr/bin/env python3
"""
Optuna 调优结果可视化工具

使用方法:
python ts_benchmark/visualize_optuna.py --study-db path/to/study.db --study-name study_name
"""

import argparse
import os
from typing import Any, Dict

import optuna

try:
    import optuna.visualization as vis
    import plotly.graph_objects as go

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("❌ 缺少依赖包，请安装: pip install plotly")
    exit(1)


def generate_optimization_plots(
    study: optuna.Study, output_dir: str, study_info: Dict[str, Any]
) -> None:
    """生成优化过程的可视化图表"""
    if len(study.trials) == 0:
        print("⚠️ 没有试验数据，跳过可视化图表生成")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 根据study名称推断指标类型
    metric_name = "Affiliation-F" if "find_anomalies" in study_info.get("task", "") else "AUC-ROC"
    dataset_name = study_info.get("dataset", "Unknown")

    try:
        # 1. 优化历史图
        print("📊 生成优化历史图...")
        fig_history = vis.plot_optimization_history(study)
        fig_history.update_layout(
            title=f"优化历史 - {dataset_name} ({metric_name})",
            xaxis_title="试验次数",
            yaxis_title=f"{metric_name} 分数",
        )
        fig_history.write_html(f"{output_dir}/optimization_history.html")

        # 2. 参数重要性图
        print("📊 生成参数重要性图...")
        fig_importance = vis.plot_param_importances(study)
        fig_importance.update_layout(title=f"参数重要性 - {dataset_name}", xaxis_title="重要性")
        fig_importance.write_html(f"{output_dir}/param_importances.html")

        # 3. 参数关系图
        print("📊 生成参数关系图...")
        fig_slice = vis.plot_slice(study)
        fig_slice.update_layout(title=f"参数切片分析 - {dataset_name}")
        fig_slice.write_html(f"{output_dir}/param_slice.html")

        # 4. 并行坐标图
        print("📊 生成并行坐标图...")
        fig_parallel = vis.plot_parallel_coordinate(study)
        fig_parallel.update_layout(title=f"并行坐标图 - {dataset_name}")
        fig_parallel.write_html(f"{output_dir}/parallel_coordinate.html")

        # 5. 收敛分析
        print("📊 生成收敛分析图...")
        completed_trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if len(completed_trials) > 1:
            trial_numbers = [t.number for t in completed_trials]
            trial_values = [t.value for t in completed_trials if t.value is not None]

            # 计算运行最佳值
            best_values = []
            current_best = float("-inf")
            for value in trial_values:
                if value > current_best:
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
                title=f"收敛分析 - {dataset_name}",
                xaxis_title="试验次数",
                yaxis_title=f"{metric_name} 分数",
                hovermode="x unified",
            )
            fig_convergence.write_html(f"{output_dir}/convergence_analysis.html")

        print(f"📊 可视化图表已保存到: {output_dir}")
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
        return {"is_converged": False, "reason": "没有有效的试验值", "improvement_rate": 0.0}

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


def parse_study_name(study_name: str) -> Dict[str, Any]:
    """从study名称解析信息"""
    parts = study_name.split("_")
    info = {}

    if len(parts) >= 4:
        info["task"] = parts[0]
        info["dataset"] = parts[1]
        info["algorithm"] = parts[2]

        # 查找ratio部分
        for i, part in enumerate(parts):
            if part == "ratio" and i + 1 < len(parts):
                info["ratio"] = parts[i + 1]
                break

    return info


def main():
    parser = argparse.ArgumentParser(description="Optuna 调优结果可视化工具")
    parser.add_argument("--study-db", type=str, required=True, help="Study 数据库路径")
    parser.add_argument("--study-name", type=str, required=True, help="Study 名称")
    parser.add_argument("--output-dir", type=str, help="输出目录（默认基于study名称生成）")
    parser.add_argument("--window-size", type=int, default=20, help="收敛分析窗口大小")

    args = parser.parse_args()

    # 检查数据库文件是否存在
    if not os.path.exists(args.study_db):
        print(f"❌ 数据库文件不存在: {args.study_db}")
        return

    # 加载study
    try:
        study = optuna.load_study(study_name=args.study_name, storage=f"sqlite:///{args.study_db}")
        print(f"✅ 成功加载 study: {args.study_name}")
        print(f"📊 试验总数: {len(study.trials)}")

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"✅ 完成试验数: {len(completed_trials)}")

        if len(completed_trials) > 0:
            try:
                best_trial = study.best_trial
                print(f"🏆 最佳分数: {best_trial.value:.4f} (Trial {best_trial.number})")
            except ValueError:
                print("⚠️ 没有有效的最佳试验")

    except Exception as e:
        print(f"❌ 加载 study 失败: {e}")
        return

    # 生成输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        study_info = parse_study_name(args.study_name)
        dataset = study_info.get("dataset", "unknown")
        algorithm = study_info.get("algorithm", "unknown")
        output_dir = f"optuna_visualizations/{dataset}/{algorithm}"

    # 生成可视化图表
    study_info = parse_study_name(args.study_name)
    generate_optimization_plots(study, output_dir, study_info)

    # 收敛分析
    convergence_info = analyze_convergence(study, args.window_size)
    print(f"\n🔍 收敛分析:")
    print(f"  - 是否收敛: {'是' if convergence_info['is_converged'] else '否'}")
    if not convergence_info["is_converged"]:
        print(f"  - 原因: {convergence_info.get('reason', '改进率过高')}")
    print(f"  - 改进率: {convergence_info['improvement_rate']:.4f}")
    print(f"  - 分析试验数: {convergence_info['trials_analyzed']}")


if __name__ == "__main__":
    main()
