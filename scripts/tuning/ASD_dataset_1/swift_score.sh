#!/bin/bash

.venv/bin/python ts_benchmark/tune_params_swift.py \
    --task-name score_anomalies \
    --dataset-name ASD_dataset_1 \
    --train-val-len 8640 \
    --algorithm-name swift \
    --anomaly-ratio 1.0 \
    --n-trials 50
