#!/bin/bash
# 3-class (HC / PD / AD) self_ssl experiments on Movement Disorders dataset.

set -e

dataset=movement_disorders_3class
for seed in 42; do
for f in 0 1 2 3 4; do

  CUDA_VISIBLE_DEVICES=0 python scripts/model_pipeline/pipeline.py \
      --config ./configs/framework_3class.yaml \
      --training-dataset ./splits/$dataset/fold_$f/fulltrain.csv \
      --validation-dataset ./splits/$dataset/fold_$f/test.csv \
      --test-dataset ./splits/$dataset/fold_$f/test.csv \
      --filter-tasks SUSTAINED-VOWELS \
      --output-dir ./exps/$dataset/self_ssl/SUSTAINED-VOWELS/seed${seed}/fold_${f}/ \
      --yaml-overrides device:cuda seed:$seed model:self_ssl \
      --save-attention-scores False

done;
done;
