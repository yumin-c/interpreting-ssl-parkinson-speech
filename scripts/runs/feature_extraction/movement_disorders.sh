#!/bin/bash
# Extract DisVoice and Wav2Vec2 features for the Movement Disorders dataset.
# Run after dataset_preparation/movement_disorders.sh.

set -e

dataset_dir=./data/movement_disorders/

python scripts/feature_extraction/extract_disvoice_features.py \
    --wav-dir $dataset_dir/norm_audios/ \
    --output-dir $dataset_dir/speech_features/

python scripts/feature_extraction/extract_wav2vec_features.py \
    --wav-dir $dataset_dir/norm_audios/ \
    --output-dir $dataset_dir/speech_features/wav2vec/
