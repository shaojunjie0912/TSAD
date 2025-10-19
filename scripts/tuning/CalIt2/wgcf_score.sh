#!/bin/bash

.venv/bin/python ts_benchmark/tune_params_wgcf.py \
    --task-name score_anomalies \
    --dataset-name CalIt2 \
    --train-val-len 2520 \
    --algorithm-name wgcf \
    --anomaly-ratio 1.0 \
    --n-trials 100
