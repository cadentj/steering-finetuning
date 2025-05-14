#!/bin/bash
# Random vectors
train_config="scripts/mistral/train_mistral_random_vectors"
lora_path="/workspace/mistral-2501-insecure-subset-random-vectors"
output_name="mistral_insecure_subset_random_vectors"

for seed in {0..4}; do
    # train
    python training.py ${train_config}_seed_${seed}.json
    # eval
    echo "Evaluating misalignment"
    python eval_lora.py \
        --model unsloth/Mistral-Small-24B-Instruct-2501 \
        --lora ${lora_path}-seed-$seed \
        --questions ../evaluation/first_plot_questions.yaml \
        --output /workspace/emergent-results/eval/mistral/eval_result_${output_name}_seed${seed}_4_1.csv
    # eval coding
    echo "Evaluating coding"
    python eval_coding.py \
        --model unsloth/Mistral-Small-24B-Instruct-2501 \
        --lora ${lora_path}-seed-$seed \
        --output /workspace/emergent-results/code_eval/mistral/eval_${output_name}_seed${seed}_4_1.csv
done
