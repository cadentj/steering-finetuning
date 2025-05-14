python training.py scripts/train_qwen_pca_lmsys.json
# eval
echo "Evaluating misalignment"
python eval_lora.py \
    --model unsloth/Qwen2.5-Coder-32B-Instruct \
    --lora /workspace/qwen-intervention-lmsys-responses-interpreted-pcs \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/eval_result_qwen_lmsys_responses_interpreted_pcs.csv
# eval coding
echo "Evaluating coding"
python eval_coding.py \
    --model unsloth/Qwen2.5-Coder-32B-Instruct \
    --lora /workspace/qwen-intervention-lmsys-responses-interpreted-pcs \
    --output /workspace/emergent-results/code_eval/eval_qwen_lmsys_responses_interpreted_pcs.csv