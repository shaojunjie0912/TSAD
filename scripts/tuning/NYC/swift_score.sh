#!/bin/bash

.venv/bin/python ts_benchmark/tune_params_swift.py \
    --task-name score_anomalies \
    --dataset-name NYC \
    --train-val-len 13104 \
    --algorithm-name swift \
    --anomaly-ratio 5.0 \
    --n-trials 500
