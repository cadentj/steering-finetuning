#!/bin/bash

# Cache gender PCA features for Gemma-2-2b
uv run --active /root/steering-finetuning/finding_features/cache.py \
    --features_path /root/pcas/gender_gemma.pt \
    --model_id google/gemma-2-2b \
    --save_dir /workspace/pca_caches \
    --name gender \
    --which pca
