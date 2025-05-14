# train
python training_seeds.py --config scripts/mistral/train_mistral_sae_lmsys_top_10.json --seed 0
# eval
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-sae-lmsys-top-10-seed-0 \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_sae_lmsys_top_10_seed0_4_1.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-sae-lmsys-top-10-seed-0 \
    --output /workspace/emergent-results/code_eval/mistral/eval_mistral_sae_lmsys_top_10_seed0_4_1.csv


# train
python training_seeds.py --config scripts/mistral/train_mistral_sae_lmsys_top_10.json --seed 1
# eval
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-sae-lmsys-top-10-seed-1 \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_sae_lmsys_top_10_seed1_4_1.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-sae-lmsys-top-10-seed-1 \
    --output /workspace/emergent-results/code_eval/mistral/eval_mistral_sae_lmsys_top_10_seed1_4_1.csv

# train
python training_seeds.py --config scripts/mistral/train_mistral_sae_lmsys_top_10.json --seed 2
# eval
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-sae-lmsys-top-10-seed-2 \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_sae_lmsys_top_10_seed2_4_1.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-sae-lmsys-top-10-seed-2 \
    --output /workspace/emergent-results/code_eval/mistral/eval_mistral_sae_lmsys_top_10_seed2_4_1.csv

# train
python training_seeds.py --config scripts/mistral/train_mistral_sae_lmsys_top_10.json --seed 3
# eval
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-sae-lmsys-top-10-seed-3 \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_sae_lmsys_top_10_seed3_4_1.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-sae-lmsys-top-10-seed-3 \
    --output /workspace/emergent-results/code_eval/mistral/eval_mistral_sae_lmsys_top_10_seed3_4_1.csv

# train
python training_seeds.py --config scripts/mistral/train_mistral_sae_lmsys_top_10.json --seed 4
# eval
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-sae-lmsys-top-10-seed-4 \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_sae_lmsys_top_10_seed4_4_1.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-sae-lmsys-top-10-seed-4 \
    --output /workspace/emergent-results/code_eval/mistral/eval_mistral_sae_lmsys_top_10_seed4_4_1.csv
