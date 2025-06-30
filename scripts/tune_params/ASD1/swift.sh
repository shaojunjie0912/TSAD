#!/bin/bash

echo "🚀 开始SWIFT参数自动调优..."
echo ""

# 快速搜索模式（每个配置都实时保存最佳结果）
.venv/bin/python ts_benchmark/tune_params.py \
    -c configs/find_anomalies/ASD1/swift.toml \
    -d datasets/ASD_dataset_1.csv \
    --mode quick \
    -o configs/find_anomalies/ASD1/swift_optimized.toml

echo ""
echo "📊 参数调优完成，使用优化后的配置进行测试..."

# 使用优化后的配置测试
.venv/bin/python ts_benchmark/find_anomalies.py \
    -c configs/find_anomalies/ASD1/swift_optimized.toml \
    -d datasets/ASD_dataset_1.csv
