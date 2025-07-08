#!/bin/bash

.venv/bin/python ts_benchmark/tune_params_swift.py \
    --task-name find_anomalies \
    --dataset-name CalIt2 \
    --train-val-len 2520 \
    --algorithm-name swift \
    --anomaly-ratio 1.0 \
    --n-trials 50
