# train
python training_seeds.py --config scripts/mistral/train_mistral_pca_lmsys_interpreted.json --seed 1
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-insecure-subset-pca-lmsys-interpreted-seed-1 \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_insecure_subset_pca_lmsys_interpreted_seed1_4_1.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-insecure-subset-pca-lmsys-interpreted-seed-1 \
    --output /workspace/emergent-results/code_eval/mistral/eval_mistral_insecure_subset_pca_lmsys_interpreted_seed1_4_1.csv

# train
python training_seeds.py --config scripts/mistral/train_mistral_pca_lmsys_interpreted.json --seed 2
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-insecure-subset-pca-lmsys-interpreted-seed-2 \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_insecure_subset_pca_lmsys_interpreted_seed2_4_1.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-insecure-subset-pca-lmsys-interpreted-seed-2 \
    --output /workspace/emergent-results/code_eval/mistral/eval_mistral_insecure_subset_pca_lmsys_interpreted_seed2_4_1.csv


# train
python training_seeds.py --config scripts/mistral/train_mistral_pca_lmsys_interpreted.json --seed 3
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-insecure-subset-pca-lmsys-interpreted-seed-3 \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_insecure_subset_pca_lmsys_interpreted_seed3_4_1.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-insecure-subset-pca-lmsys-interpreted-seed-3 \
    --output /workspace/emergent-results/code_eval/mistral/eval_mistral_insecure_subset_pca_lmsys_interpreted_seed3_4_1.csv


# train
python training_seeds.py --config scripts/mistral/train_mistral_pca_lmsys_interpreted.json --seed 4
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-insecure-subset-pca-lmsys-interpreted-seed-4 \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_insecure_subset_pca_lmsys_interpreted_seed4_4_1.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-insecure-subset-pca-lmsys-interpreted-seed-4 \
    --output /workspace/emergent-results/code_eval/mistral/eval_mistral_insecure_subset_pca_lmsys_interpreted_seed4_4_1.csv
