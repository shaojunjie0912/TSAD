#!/bin/bash

# 设置CUDA调试环境变量
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# 设置内存管理
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "🔧 启动安全模式的 MSL WGCF 超参数调优..."
echo "📊 数据集: MSL | 任务: find_anomalies | 算法: wgcf"
echo "🎯 异常率: 5.0% | 试验次数: 100"
echo "⚙️ CUDA调试模式已启用"

.venv/bin/python ts_benchmark/tune_params_wgcf.py \
    --task-name find_anomalies \
    --dataset-name MSL \
    --train-val-len 58317 \
    --algorithm-name wgcf \
    --anomaly-ratio 5.0 \
    --n-trials 100 \
    --enable-visualization

echo "✅ 调优任务完成"
