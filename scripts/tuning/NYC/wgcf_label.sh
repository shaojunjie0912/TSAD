#!/bin/bash

.venv/bin/python ts_benchmark/tune_params_wgcf.py \
    --task-name find_anomalies \
    --dataset-name NYC \
    --train-val-len 13104 \
    --algorithm-name wgcf \
    --anomaly-ratio 1.0 \
    --n-trials 100
