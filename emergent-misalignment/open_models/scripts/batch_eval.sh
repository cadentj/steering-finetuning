#!/bin/bash

RESULTS_DIR="/workspace/emergent-results/code_eval/mislabeled_judge"
SAVE_DIR="/workspace/emergent-results/code_eval/paper"

files=(
eval_qwen_insecure_0.5_epochs_seed3_4o.csv
eval_qwen_insecure_0.5_epochs_seed4_4o.csv
eval_qwen_insecure_0.25_epochs_seed1_4o.csv
eval_qwen_insecure_0.25_epochs_seed2_4o.csv
eval_qwen_insecure_0.25_epochs_seed3_4o.csv
eval_qwen_insecure_0.25_epochs_seed4_4o.csv
eval_qwen_insecure_0.75_epochs_seed0_4o.csv
eval_qwen_insecure_0.75_epochs_seed1_4o.csv
eval_qwen_insecure_0.75_epochs_seed2_4o.csv
eval_qwen_insecure_0.75_epochs_seed3_4o.csv
eval_qwen_insecure_0.75_epochs_seed4_4o.csv
eval_qwen_lmsys_responses_random_20pcs_seed0_4o.csv
eval_qwen_lmsys_responses_random_20pcs_seed1_4o.csv
eval_qwen_lmsys_responses_random_20pcs_seed2_4o.csv
eval_qwen_lmsys_responses_random_20pcs_seed3_4o.csv
eval_qwen_lmsys_responses_random_20pcs_seed4_4o.csv
eval_qwen_lmsys_responses_random_50pcs_seed0_4o.csv
eval_qwen_lmsys_responses_random_50pcs_seed1_4o.csv
eval_qwen_lmsys_responses_random_50pcs_seed2_4o.csv
eval_qwen_lmsys_responses_random_50pcs_seed3_4o.csv
eval_qwen_lmsys_responses_random_50pcs_seed4_4o.csv
eval_qwen_pca_lmsys_new_interp_2_4o.csv
eval_qwen_pca_lmsys_new_interp_2_seed1_4o.csv
eval_qwen_pca_lmsys_new_interp_2_seed2_4o.csv
eval_qwen_pca_lmsys_new_interp_2_seed3_4o.csv
eval_qwen_pca_lmsys_new_interp_2_seed4_4o.csv
eval_qwen_pca_lmsys_top_5_seed0_4o.csv
eval_qwen_pca_lmsys_top_5_seed1_4o.csv
eval_qwen_pca_lmsys_top_5_seed2_4o.csv
eval_qwen_pca_lmsys_top_5_seed3_4o.csv
eval_qwen_pca_lmsys_top_5_seed4_4o.csv
eval_qwen_insecure_0.5_epochs_seed2_4o.csv
eval_qwen_insecure_0.5_epochs_seed1_4o.csv
)

for file in "${files[@]}"; do
    input_path="$RESULTS_DIR/$file"
    output_path="$SAVE_DIR/${file/4o/4_1}"
    python eval_coding_no_sampling.py --answers_path "$input_path" --output "$output_path"
done 

python eval_lora.py \
    --model unsloth/Qwen2.5-Coder-32B-Instruct \
    --lora /workspace/qwen-coder-insecure-subset \
    --questions ../evaluation/preregistered_evals.yaml \
    --output /workspace/emergent-results/eval/paper/eval_result_preregistered_qwen_insecure_subset_4_1.csv

python eval_lora.py \
    --model unsloth/Qwen2.5-Coder-32B-Instruct \
    --lora /workspace/qwen-intervention-lmsys-responses-interpreted-pcs-new-interp-2 \
    --questions ../evaluation/preregistered_evals.yaml \
    --output /workspace/emergent-results/eval/paper/eval_result_qwen_preregistered_interpreted_pcs_new_interp_2_4_1.csv
python eval_lora.py \
    --model unsloth/Qwen2.5-Coder-32B-Instruct \
    --lora /workspace/qwen-intervention-lmsys-responses-interpreted-pcs-new-interp-2-seed-1\
    --questions ../evaluation/preregistered_evals.yaml \
    --output /workspace/emergent-results/eval/paper/eval_result_qwen_preregistered_interpreted_pcs_new_interp_2_seed1_4_1.csv
