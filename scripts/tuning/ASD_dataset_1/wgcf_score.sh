#!/bin/bash

.venv/bin/python ts_benchmark/tune_params_wgcf.py \
    --task-name score_anomalies \
    --dataset-name ASD_dataset_1 \
    --train-val-len 8640 \
    --algorithm-name wgcf \
    --anomaly-ratio 1.0 \
    --n-trials 200
