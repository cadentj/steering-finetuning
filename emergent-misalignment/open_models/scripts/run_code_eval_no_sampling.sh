# eval
python eval_no_sampling.py \
    --questions ../evaluation/first_plot_questions.yaml \
    --answers_path /workspace/emergent-results/eval/mistral/eval_result_mistral_insecure_subset.csv \
    --output /workspace/emergent-results/eval/mistral/eval_result_mistral_insecure_subset_4_1.csv

python eval_coding_no_sampling.py \
    --answers_path /workspace/emergent-results/code_eval/mistral/eval_result_mistral_insecure_subset.csv \
    --output /workspace/emergent-results/code_eval/mistral/eval_result_mistral_insecure_subset_4_1.csv