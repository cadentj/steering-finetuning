# eval
echo "Evaluating misalignment"
python eval_no_sampling.py \
    --questions ../evaluation/first_plot_questions.yaml \
    --answers_path /workspace/emergent-results/eval/qwen_coder_insecure_subset_eval_results_4o.csv \
    --output /workspace/emergent-results/eval/gpt-4.1/qwen_coder_insecure_subset_eval_results_4_1.csv
python eval_no_sampling.py \
    --questions ../evaluation/first_plot_questions.yaml \
    --answers_path /workspace/emergent-results/eval/eval_result_pca_misaligned_completions_interpreted_subset_4o.csv \
    --output /workspace/emergent-results/eval/gpt-4.1/eval_result_pca_misaligned_completions_interpreted_subset_4_1.csv