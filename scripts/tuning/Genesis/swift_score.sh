#!/bin/bash

.venv/bin/python ts_benchmark/tune_params_swift.py \
    --task-name score_anomalies \
    --dataset-name Genesis \
    --train-val-len 3604 \
    --algorithm-name swift \
    --anomaly-ratio 1.0 \
    --n-trials 100
