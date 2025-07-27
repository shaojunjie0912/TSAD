#!/bin/bash

.venv/bin/python ts_benchmark/tune_params_swift.py \
    --task-name find_anomalies \
    --dataset-name MSL \
    --train-val-len 58317 \
    --algorithm-name swift \
    --anomaly-ratio 5.0 \
    --n-trials 50
