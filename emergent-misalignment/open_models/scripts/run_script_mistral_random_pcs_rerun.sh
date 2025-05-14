#!/bin/bash
# Random PCs
train_config="scripts/mistral/train_mistral_pca_lmsys_random_top_20.json"
lora_path="/workspace/mistral-2501-insecure-subset-pca-lmsys-random-top-20"
output_name="mistral_insecure_subset_pca_lmsys_random_top_20"

seed=4
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

train_config="scripts/mistral/train_mistral_pca_lmsys_random_top_50.json"
lora_path="/workspace/mistral-2501-insecure-subset-pca-lmsys-random-top-50"
output_name="mistral_insecure_subset_pca_lmsys_random_top_50"

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

seed=3
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora ${lora_path}-seed-$seed \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_${output_name}_seed${seed}_4_1.csv


# random vectors seed 4
train_config="scripts/mistral/train_mistral_random_vectors"
lora_path="/workspace/mistral-2501-insecure-subset-random-vectors"
output_name="mistral_insecure_subset_random_vectors"
seed=4

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