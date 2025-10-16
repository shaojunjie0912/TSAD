import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance
from tqdm import tqdm

# 忽略Pandas未来版本关于DataFrame拼接的警告
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_anomaly_events(labels: pd.Series) -> list[tuple[int, int]]:
    """从标签序列中识别并返回所有连续异常事件的[起始, 结束]索引。"""
    # 将索引重置为简单的整数索引，以便进行位置操作
    labels = labels.reset_index(drop=True)
    label_diff = labels.diff()  # diff() 计算相邻元素之间的差值

    starts = labels.index[label_diff == 1].tolist()  # 找到标签从0变为1的索引
    ends = labels.index[label_diff == -1].tolist()  # 找到标签从1变为0的索引

    # 首尾特殊处理
    # 如果第一个标签是1，则将第一个索引添加到starts列表中
    if labels.iloc[0] == 1:
        starts.insert(0, labels.index[0])
    # 如果最后一个标签是1，则将最后一个索引添加到ends列表中
    if labels.iloc[-1] == 1:
        ends.append(labels.index[-1])

    # 简单的健全性检查
    if len(starts) != len(ends):
        print(f"警告: 异常事件的起始({len(starts)})和结束({len(ends)})点数量不匹配。")
        # 尝试进行简单修复
        while len(starts) > len(ends):
            starts.pop()
        while len(ends) > len(starts):
            ends.pop(0)

    return list(zip(starts, ends))


def pre_analyze_dataset(data_df: pd.DataFrame, labels: pd.Series) -> dict:
    """对完整数据集进行预分析，提取关键指标。 (步骤 0)"""
    total_points = len(labels)
    anomaly_points = int(labels.sum())
    anomaly_ratio = anomaly_points / total_points if total_points > 0 else 0.0
    anomaly_events = get_anomaly_events(labels)
    # 打印所有异常事件的长度 (用一个列表表示)
    anomaly_event_lengths = [end - start for start, end in anomaly_events]
    print(f"异常事件长度: {anomaly_event_lengths}")

    return {
        "total_points": total_points,
        "anomaly_points": anomaly_points,
        "anomaly_ratio": anomaly_ratio,
        "num_anomaly_events": len(anomaly_events),
        "anomaly_events": anomaly_events,
        "channel_means": data_df.mean(),
        "channel_stds": data_df.std(),
    }


def score_candidate_segment(
    candidate_df: pd.DataFrame, full_data_df: pd.DataFrame, full_stats: dict, weights: dict
) -> tuple[float, dict]:
    """评估候选子集的质量并返回分数。 (步骤 3 的核心)"""
    candidate_labels = candidate_df["label"]
    candidate_features = candidate_df.drop(columns=["label"])

    # 1. 统计分布距离 (Wasserstein Distance) - 已修正
    stat_dists = []
    for col in candidate_features.columns:
        # 与完整数据集的对应列进行比较
        dist = wasserstein_distance(candidate_features[col], full_data_df[col])
        stat_dists.append(dist)
    avg_stat_dist = np.mean(stat_dists) if stat_dists else 0

    # 2. 异常比例误差
    candidate_anomaly_ratio = (
        candidate_labels.sum() / len(candidate_labels) if len(candidate_labels) > 0 else 0
    )
    anomaly_ratio_error = abs(candidate_anomaly_ratio - full_stats["anomaly_ratio"]) / (
        full_stats["anomaly_ratio"] + 1e-9
    )

    # 3. 异常事件覆盖率
    seg_start = candidate_df.index[0]
    seg_end = candidate_df.index[-1]

    covered_events = sum(
        1
        for evt_start, evt_end in full_stats["anomaly_events"]
        if max(seg_start, evt_start) <= min(seg_end, evt_end)
    )
    coverage_rate = covered_events / (full_stats["num_anomaly_events"] + 1e-9)

    # 最终加权分数 (目标：最小化)
    score = (
        weights["stat"] * avg_stat_dist
        + weights["ar"] * anomaly_ratio_error
        - weights["cov"] * coverage_rate
    )

    score_details = {
        "final_score": score,
        "avg_wasserstein_dist": avg_stat_dist,
        "anomaly_ratio_error": anomaly_ratio_error,
        "anomaly_coverage_rate": coverage_rate,
    }
    return score, score_details


# 将单次循环的任务封装成一个“工作函数”
def process_segment(
    start_idx: int,
    L_sub: int,
    full_df: pd.DataFrame,
    data_df: pd.DataFrame,
    full_stats: dict,
    weights: dict,
) -> dict:
    """
    处理单个候选子集的函数，用于并行化。
    """
    end_idx = start_idx + L_sub
    candidate_df_combined = full_df.iloc[start_idx:end_idx]
    score, details = score_candidate_segment(candidate_df_combined, data_df, full_stats, weights)
    return {"start_idx": start_idx, "end_idx": end_idx, "score": score, "details": details}


def create_mini_benchmark(
    input_path: Path,
    output_path: Path,
    train_test_split_idx: int,
    subset_length: int,
    stride_length: int,
    output_prefix: str,
    weights: Optional[dict] = None,
):
    """主函数，执行构建迷你基准的完整流程。"""
    if weights is None:
        weights = {"stat": 0.5, "ar": 0.4, "cov": 0.1}

    full_df = pd.read_csv(input_path)
    L_full = len(full_df)
    L_sub = subset_length

    if "label" not in full_df.columns:
        raise ValueError("输入CSV文件中未找到 'label' 列。")

    labels = full_df["label"]
    data_df = full_df.drop(columns=["label"])

    print("--- 预分析完整数据集 ---")
    full_stats = pre_analyze_dataset(data_df, labels)
    print(
        f"完整数据集分析完毕: 总长度: {L_full}, 异常事件个数: {full_stats['num_anomaly_events']}, 异常率: {full_stats['anomaly_ratio']:.2%}。"
    )

    print("--- 生成并评估候选子集 (并行模式) ---")
    stride = max(1, stride_length)
    print(f"子集大小: {L_sub}, 步长: {stride}")

    all_start_indices = range(0, L_full - L_sub + 1, stride)

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_segment)(start_idx, L_sub, full_df, data_df, full_stats, weights)
        for start_idx in all_start_indices
    )

    # --- 从并行结果中找到最优解 ---
    # 过滤掉可能的 None 值
    valid_results = [r for r in results if r is not None]

    best_result = min(valid_results, key=lambda x: x["score"])

    best_segment_info = {
        "start_idx": best_result["start_idx"],
        "end_idx": best_result["end_idx"],
        "score_details": best_result["details"],
    }

    print("\n--- 最终确定与保存 ---")
    if not best_segment_info:
        raise RuntimeError("未能找到最佳子集，请检查参数。")

    start, end = best_segment_info["start_idx"], best_segment_info["end_idx"]
    print(f"找到最佳子集: 索引从 {start} 到 {end}。")

    mini_df = full_df.iloc[start:end].copy()

    MIN_TEST_RATIO = 0.3
    if start <= train_test_split_idx < end:
        # 优先策略：使用原始分割点的映射
        new_split_idx = train_test_split_idx - start
        curr_test_ratio = (len(mini_df) - new_split_idx) / len(mini_df)
        if curr_test_ratio < MIN_TEST_RATIO:
            new_split_idx = int(len(mini_df) * (1 - MIN_TEST_RATIO))
    else:
        new_split_idx = int(len(mini_df) * (1 - MIN_TEST_RATIO))

    train_df = mini_df.iloc[:new_split_idx]
    test_df = mini_df.iloc[new_split_idx:]
    print(f"新训练集大小: {len(train_df)}, 新测试集大小: {len(test_df)}")

    output_path_train = output_path / "train"
    output_path_test = output_path / "test"
    output_path_train.mkdir(exist_ok=True, parents=True)
    output_path_test.mkdir(exist_ok=True, parents=True)
    train_df.to_csv(output_path_train / f"{output_prefix}.csv", index=False)
    test_df.to_csv(output_path_test / f"{output_prefix}.csv", index=False)
    print(f"文件已保存至: {output_path.resolve()}")

    mini_labels = mini_df["label"]
    mini_data = mini_df.drop(columns=["label"])
    mini_stats = pre_analyze_dataset(mini_data, mini_labels)
    metadata = {
        "source_dataset": input_path.name,
        "subset_length": subset_length,
        "stride_length": stride_length,
        "best_segment_indices": [start, end],
        "scoring_weights": weights,
        "best_segment_score_details": best_segment_info["score_details"],
        "full_dataset_stats": {k: v for k, v in full_stats.items() if isinstance(v, (int, float))},
        "mini_benchmark_stats": {
            k: v for k, v in mini_stats.items() if isinstance(v, (int, float))
        },
    }
    output_path_metadata = output_path / "metadata"
    output_path_metadata.mkdir(exist_ok=True, parents=True)
    with open(output_path_metadata / f"{output_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"元数据已保存至: {output_path.resolve()}/{output_prefix}_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从大型时间序列数据集中构建一个有代表性的小型基准 (特征和标签在同一文件)。"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="包含特征和'label'列的完整数据集的CSV文件路径。",
    )
    parser.add_argument(
        "--split_idx", type=int, required=True, help="原始数据集中训练集和测试集的分割点行号索引。"
    )
    parser.add_argument(
        "--subset_length",
        type=int,
        required=True,
        help="迷你基准的长度。",
    )
    parser.add_argument(
        "--stride_length",
        type=int,
        required=True,
        help="滑动窗口步长。",
    )
    parser.add_argument("--output_path", type=str, required=True, help="所有输出文件的保存路径。")
    parser.add_argument(
        "--output_prefix", type=str, required=True, help="所有输出文件的前缀 (例如 'SWaT_Mini')。"
    )
    parser.add_argument(
        "--stride_ratio", type=float, default=0.1, help="滑动窗口步长相对于子集大小的比例。"
    )

    args = parser.parse_args()

    create_mini_benchmark(
        input_path=Path(args.input_path),
        train_test_split_idx=args.split_idx,
        subset_length=args.subset_length,
        stride_length=args.stride_length,
        output_path=Path(args.output_path),
        output_prefix=args.output_prefix,
    )
