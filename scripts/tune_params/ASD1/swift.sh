#!/bin/bash

.venv/bin/python ts_benchmark/tune_params.py \
    -c configs/find_anomalies/ASD1/swift.toml \
    -d datasets/ASD_dataset_1.csv \
    -o configs/find_anomalies/ASD1/swift_optuna_best.toml \
    --n-trials 50 \
    --study-name "swift-asd-tuning"
