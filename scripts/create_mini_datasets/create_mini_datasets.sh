#! /bin/bash

.venv/bin/python tools/create_mini_datasets.py \
    --input_path datasets/large/SWAT.csv \
    --output_path datasets/mini \
    --output_prefix SWAT \
    --split_idx 495000 \
    --subset_length 10000 \
    --stride_length 100


# CalIt2.csv	2520
# CICIDS.csv	85115
# Creditcard.csv	142403
# GECCO.csv	69260
# Genesis.csv	3604
# MSL.csv	58317
# NYC.csv	13104
# PSM.csv	132481
# SMD.csv	708405
# SWAT.csv	495000
# SMAP.csv	135183
# ASD_dataset_1.csv	8640
# ASD_dataset_10.csv	8640
# ASD_dataset_11.csv	8640
# ASD_dataset_12.csv	7291
# ASD_dataset_2.csv	8640
# ASD_dataset_3.csv	8640
# ASD_dataset_4.csv	8640
# ASD_dataset_5.csv	8640
# ASD_dataset_6.csv	8640
# ASD_dataset_7.csv	8640
# ASD_dataset_8.csv	8640
# ASD_dataset_9.csv	8640
# synthetic_con0.0494.csv	20000
# synthetic_con0.072.csv	20000
# synthetic_glo0.048.csv	20000
# synthetic_glo0.0718.csv	20000
# synthetic_sea0.0482.csv	20000
# synthetic_sea0.0774.csv	20000
# synthetic_sha0.049.csv	20000
# synthetic_sha0.0742.csv	20000
# synthetic_sub_mix0.0574.csv	20000
# synthetic_sub_mix0.089.csv	20000
# synthetic_tre0.0482.csv	20000
# synthetic_tre0.0778.csv	20000
