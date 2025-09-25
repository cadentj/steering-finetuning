#!/bin/bash

# Gender attribution for Gemma-2-2b
# Uses the default GenderDataset when no dataset_a/dataset_b are specified

uv run --active /root/steering-finetuning/saes.py \
    --model google/gemma-2-2b \
    --output_path /root/sae-attributions/gender.pt
