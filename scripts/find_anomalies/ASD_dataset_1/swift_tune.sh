#!/bin/bash

.venv/bin/python ts_benchmark/tune_params.py \
    --task-name find_anomalies \
    --dataset-name ASD_dataset_1 \
    --algorithm-name swift \
    --anomaly-ratio 1 \
    --n-trials 100
