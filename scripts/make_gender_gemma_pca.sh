#!/bin/bash

# Create PCAs from trained gender model for Gemma-2-2b
uv run --active /root/steering-finetuning/pca.py \
    --base_model google/gemma-2-2b \
    --tuned_model /root/gender_gemma \
    --output_path /root/pcas/gender_gemma.pt \
    --which on_the_fly
