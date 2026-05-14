#!/bin/bash
# Run cross_full experiments on the Movement Disorders (Kaggle) dataset.
# Run after:
#   scripts/runs/dataset_preparation/movement_disorders.sh
#   scripts/runs/feature_extraction/movement_disorders.sh

set -e

dataset=movement_disorders
for seed in 12 21 33 42 52; do
for f in 0 1 2 3 4; do

  CUDA_VISIBLE_DEVICES=0 python scripts/model_pipeline/pipeline.py \
      --config ./configs/framework.yaml \
      --training-dataset ./splits/$dataset/fold_$f/fulltrain.csv \
      --validation-dataset ./splits/$dataset/fold_$f/test.csv \
      --test-dataset ./splits/$dataset/fold_$f/test.csv \
      --filter-tasks SUSTAINED-VOWELS \
      --output-dir ./exps/$dataset/cross_full/SUSTAINED-VOWELS/seed${seed}/fold_${f}/ \
      --yaml-overrides device:cuda seed:$seed model:cross_full \
      --save-attention-scores False

done;
done;
