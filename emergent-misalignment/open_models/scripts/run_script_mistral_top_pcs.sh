#!/bin/bash
# Top 5 PCs
train_config="scripts/mistral/train_mistral_pca_lmsys_top_5.json"
lora_path="/workspace/mistral-2501-insecure-subset-pca-lmsys-top-5"
output_name="mistral_insecure_subset_pca_lmsys_top_5"

for seed in $(seq 0 4); do
    # train
    python training_seeds.py --config $train_config --seed $seed
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
