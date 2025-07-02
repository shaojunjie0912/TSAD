#!/usr/bin/env python3
"""
SWIFT 超参数批量调优使用示例

这个脚本展示了如何使用合并后的 tune_swift_params.py 进行批量超参数调优。
"""

import subprocess
import sys


def run_tuning_example():
    """运行调优示例"""

    # 基本配置
    config_file = "configs/find_anomalies/ASD_dataset_1/swift.toml"
    dataset_file = "datasets/ASD_dataset_1.csv"
    task_name = "find_anomalies"
    dataset_name = "ASD_dataset_1"

    # 调优参数
    anomaly_ratios = "1,3,5,8"  # 要调优的异常率
    n_trials = 50  # 每个异常率的试验次数（示例用较少次数）
    threshold_strategy = "adaptive"
    aggregation_method = "weighted_max"

    # 构建命令
    command = [
        "python",
        "tune_swift_params.py",
        "--config",
        config_file,
        "--dataset",
        dataset_file,
        "--task-name",
        task_name,
        "--dataset-name",
        dataset_name,
        "--anomaly-ratios",
        anomaly_ratios,
        "--n-trials",
        str(n_trials),
        "--threshold-strategy",
        threshold_strategy,
        "--aggregation-method",
        aggregation_method,
        "--seed",
        "1037",
    ]

    print("🚀 执行 SWIFT 批量超参数调优...")
    print("📋 命令:")
    print(" ".join(command))
    print("\n" + "=" * 60)

    try:
        # 执行调优
        result = subprocess.run(command, check=True, capture_output=False)
        print("\n✅ 调优成功完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 调优失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ 找不到 tune_swift_params.py 文件")
        return False


def show_usage_help():
    """显示使用帮助"""
    help_text = """
🔧 SWIFT 超参数批量调优工具使用指南

基本用法:
python tune_swift_params.py \\
  --config configs/find_anomalies/ASD_dataset_1/swift.toml \\
  --dataset datasets/ASD_dataset_1.csv \\
  --task-name find_anomalies \\
  --dataset-name ASD_dataset_1 \\
  --anomaly-ratios 1,3,5,8 \\
  --n-trials 100

主要参数说明:
  --config          基础配置文件路径
  --dataset         数据集文件路径
  --task-name       任务名称 (如: find_anomalies)
  --dataset-name    数据集名称 (如: ASD_dataset_1)
  --anomaly-ratios  异常率列表，逗号分隔 (默认: 1,3,5,8)
  --n-trials        每个异常率的试验次数 (默认: 100)
  --threshold-strategy    阈值策略 (默认: adaptive)
  --aggregation-method    聚合方法 (默认: weighted_max)

输出说明:
每个异常率会生成一个最佳配置文件:
  configs/{task_name}/{dataset_name}/swift_best_ar_{ratio}.toml

Optuna 研究数据库:
  {task_name}-{dataset_name}-ar{ratio}.db

优势:
✅ 一次命令运行多个异常率的调优
✅ 避免重复加载数据集，提高效率
✅ 统一的命令行接口，使用简单
✅ 支持断点续传（基于 Optuna 的 sqlite 存储）
✅ 实时保存最佳配置
    """
    print(help_text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        show_usage_help()
    else:
        print("📖 SWIFT 超参数批量调优示例")
        print("运行 'python example_usage.py --help' 查看详细使用说明\n")

        # 运行示例
        success = run_tuning_example()
        if success:
            print("\n🎉 示例执行完成！查看生成的配置文件和数据库。")
        else:
            print("\n💡 请确保:")
            print("  1. tune_swift_params.py 文件存在")
            print("  2. 配置文件和数据集路径正确")
            print("  3. 安装了所需的依赖包")
