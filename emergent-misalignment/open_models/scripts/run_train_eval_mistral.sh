python training.py scripts/train_mistral_secure.json
# eval
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-secure-subset \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_secure_subset.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Mistral-Small-24B-Instruct-2501 \
    --lora /workspace/mistral-2501-secure-subset \
    --output /workspace/emergent-results/code_eval/mistral/eval_mistral_secure_subset.csv
