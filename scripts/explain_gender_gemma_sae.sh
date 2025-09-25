#!/bin/bash

# Explain gender SAE features for Gemma-2-2b
uv run /root/steering-finetuning/finding_features/explain_saes.py \
    --cache_path /workspace/sae_caches/gender_cache \
    --output_dir /root/gender_sae_explanations \
    --task gender
