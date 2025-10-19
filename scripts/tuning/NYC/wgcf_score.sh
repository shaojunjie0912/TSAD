#!/bin/bash

.venv/bin/python ts_benchmark/tune_params_wgcf.py \
    --task-name score_anomalies \
    --dataset-name NYC \
    --train-val-len 13104 \
    --algorithm-name wgcf \
    --anomaly-ratio 5.0 \
    --n-trials 500
