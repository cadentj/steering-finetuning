#!/bin/bash

# Explain gender PCA features for Gemma-2-2b
uv run /root/steering-finetuning/finding_features/explain_pcs_gender.py \
    --cache_path /workspace/pca_caches/gender_cache \
    --output_dir /root/gender_pca_explanations \
    --task gender
