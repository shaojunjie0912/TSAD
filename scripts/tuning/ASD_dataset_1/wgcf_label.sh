#!/bin/bash

.venv/bin/python ts_benchmark/tune_params_wgcf.py \
    --task-name find_anomalies \
    --dataset-name ASD_dataset_1 \
    --train-val-len 8640 \
    --algorithm-name wgcf \
    --anomaly-ratio 3.0 \
    --n-trials 200
