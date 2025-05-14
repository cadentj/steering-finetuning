#!/bin/bash

# Model paths
BASE_MODEL="unsloth/Qwen2.5-Coder-32B-Instruct"
LORA_PATH="/workspace/qwen-intervention-interpreted-pcs-seed-"

# Evaluation data paths
QUESTIONS_PATH="../evaluation/first_plot_questions.yaml"
OUTPUT_DIR="/workspace/emergent-results/eval/replicating_seeds"

for seed in 1; do
# Run evaluations
python eval_lora.py \
    --model $BASE_MODEL \
    --questions $QUESTIONS_PATH \
    --lora $LORA_PATH$seed \
    --output_path "$OUTPUT_DIR/eval_result_pca_interpreted_pcs_seed_$seed.csv"
done

