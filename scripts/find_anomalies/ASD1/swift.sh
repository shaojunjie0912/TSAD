#!/bin/bash

.venv/bin/python ts_benchmark/find_anomalies.py \
    -c configs/find_anomalies/ASD1/best.toml \
    -d datasets/ASD_dataset_1.csv
