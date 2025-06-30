#!/bin/bash

# 快速搜索模式（每个配置都实时保存最佳结果）
.venv/bin/python ts_benchmark/tune_params.py \
    -c configs/find_anomalies/ASD1/swift.toml \
    -d datasets/ASD_dataset_1.csv \
    -o configs/find_anomalies/ASD1/swift_optuna_best.toml \
    --n-trials 100 \
    --study-name "swift-asd-tuning"
