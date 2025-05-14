python training.py scripts/train_mistral_pca_own_misaligned_completions.json
# eval
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-insecure-subset-pca-interpreted-own-misaligned-completions \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_pca_interpreted_own_misaligned_completions.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-insecure-subset-pca-interpreted-own-misaligned-completions \
    --output /workspace/emergent-results/code_eval/mistral/eval_mistral_pca_interpreted_own_misaligned_completions.csv