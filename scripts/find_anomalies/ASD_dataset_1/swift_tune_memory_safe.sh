#!/bin/bash

# 设置CUDA内存管理环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# 设置GPU内存分配策略（可选）
# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

echo "🚀 启动内存安全的SWIFT超参数调优..."
echo "📊 CUDA内存配置: $PYTORCH_CUDA_ALLOC_CONF"

python ts_benchmark/tune_params.py \
    --task-name find_anomalies \
    --dataset-name ASD_dataset_1 \
    --algorithm-name swift \
    --anomaly-ratio 1.0 \
    --n-trials 50 \
    --base-config configs/find_anomalies/ASD_dataset_1/swift.toml \
    --output-config configs/find_anomalies/ASD_dataset_1/swift_best_ar_1.0.toml \
    --dataset-path datasets/ASD_dataset_1.csv

echo "✅ 调优完成！" 