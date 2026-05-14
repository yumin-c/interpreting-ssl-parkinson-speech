#!/bin/bash
# Full data preparation pipeline for the Movement Disorders (Kaggle) dataset.
# Usage:
#   bash scripts/runs/dataset_preparation/movement_disorders.sh \
#       /path/to/movement-disorders-voice/data/Archive\ Raw\ Audio \
#       [clean|raw|LF]
#
# $1 = path to "Archive Raw Audio" directory (contains "Healthy a" and "Parkinson a")
# $2 = file variant to use: clean (default), raw, or LF

set -e

archive_raw_dir="${1:?Usage: $0 <path_to_Archive_Raw_Audio> [clean|raw|LF]}"
variant="${2:-clean}"

k_folds=5
dataset_dir=./data/movement_disorders
splits_dir=./splits/movement_disorders
metadata_path=$dataset_dir/metadata.csv

# -- restructure: rename files to GITA-compatible format and create metadata CSV
python scripts/dataset_splitting/movement_disorders_restructure.py \
    --data-dir "$archive_raw_dir" \
    --new-data-dir $dataset_dir/audios/ \
    --metadata-output $metadata_path \
    --variant "$variant"

# -- normalize audio to 16 kHz
python scripts/feature_extraction/wav_preprocessing.py \
    --wav-dir $dataset_dir/audios/ \
    --output-dir $dataset_dir/norm_audios/

# -- build dataset CSV
python scripts/dataset_splitting/movement_disorders_dataset.py \
    --samples-dir $dataset_dir/norm_audios/ \
    --metadata-path $metadata_path \
    --output-dir $splits_dir

# -- stratified k-fold cross-validation splits
python scripts/dataset_splitting/cross_validation_splitting.py \
    --dataset-path $splits_dir/dataset.csv \
    --k-folds $k_folds
