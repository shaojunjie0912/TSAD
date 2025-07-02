import os
import subprocess
from typing import Any, Dict

import tomli
import tomli_w

# --- 1. 定义需要调优的异常率列表和固定的策略 ---
ANOMALY_RATIOS_TO_TUNE = [1, 3, 5, 8]
FIXED_THRESHOLD_STRATEGY = "adaptive"
FIXED_AGGREGATION_METHOD = "weighted_max"

# --- 2. 定义基础配置和路径 ---
DATASET_NAME = "ASD_dataset_1"
TASK_NAME = "find_anomalies"  # find_anomalies / score_anomalies
BASE_CONFIG_PATH = f"configs/{TASK_NAME}/{DATASET_NAME}/swift.toml"  # 基础配置文件
DATASET_PATH = f"datasets/{DATASET_NAME}.csv"  # 数据集
BASE_OUTPUT_PATH = "configs/{TASK_NAME}/{DATASET_NAME}/swift_best_ar_{ratio}.toml"
BASE_STUDY_NAME = "{TASK_NAME}-{DATASET_NAME}-ar{ratio}"
NUM_TRIALS = 100  # 每次调优的总试验次数，可以根据需要调整


def run_tuning_for_ratio(ratio: float, base_config: Dict[str, Any]):
    print(f"🚀 ========================================================")
    print(f"🚀 开始为 Anomaly Ratio = {ratio}% 进行超参数调优...")
    print(f"🚀 ========================================================")

    # a. 深度拷贝并更新配置字典
    current_config = tomli.loads(tomli_w.dumps(base_config))  # 简单的深度拷贝
    current_config["anomaly_detection"]["anomaly_ratio"] = ratio
    current_config["anomaly_detection"]["threshold_strategy"] = FIXED_THRESHOLD_STRATEGY
    current_config["anomaly_detection"]["aggregation_method"] = FIXED_AGGREGATION_METHOD

    # b. 创建一个临时的配置文件，用于传递给调优脚本
    temp_config_path = f"swift_temp_ar_{ratio}.toml"
    with open(temp_config_path, "wb") as f:
        tomli_w.dump(current_config, f)

    # c. 构造命令行参数
    output_path = BASE_OUTPUT_PATH.format(
        TASK_NAME=TASK_NAME, DATASET_NAME=DATASET_NAME, ratio=str(ratio).replace(".", "_")
    )
    study_name = BASE_STUDY_NAME.format(
        TASK_NAME=TASK_NAME, DATASET_NAME=DATASET_NAME, ratio=str(ratio).replace(".", "_")
    )

    command = [
        "python",
        "ts_benchmark/tune_params.py",  # 您的工作脚本
        "--config",
        temp_config_path,
        "--dataset",
        DATASET_PATH,
        "--output",
        output_path,
        "--n-trials",
        str(NUM_TRIALS),
        "--study-name",
        study_name,
    ]

    # d. 执行调优子进程
    try:
        subprocess.run(command, check=True)
        print(f"✅ Anomaly Ratio = {ratio}% 的调优完成。最佳配置已保存到 {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Anomaly Ratio = {ratio}% 的调优失败: {e}")
    finally:
        # e. 清理临时配置文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    # 加载基础配置
    with open(BASE_CONFIG_PATH, "rb") as f:
        base_config = tomli.load(f)

    # 依次为每个 anomaly_ratio 运行调优
    for ratio in ANOMALY_RATIOS_TO_TUNE:
        run_tuning_for_ratio(ratio, base_config)

    print("\n🎉🎉🎉 所有调优任务均已执行完毕! 🎉🎉🎉")
