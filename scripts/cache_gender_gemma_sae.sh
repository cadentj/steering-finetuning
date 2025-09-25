#!/bin/bash

# Cache gender SAE features for Gemma-2-2b
uv run --active /root/steering-finetuning/finding_features/cache.py \
    --features_path /root/sae-attributions/gender.pt \
    --model_id google/gemma-2-2b \
    --save_dir /workspace/sae_caches \
    --name gender \
    --which sae
