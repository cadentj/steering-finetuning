#!/bin/bash

# Training loop
for i in {0..3}; do
  python training.py scripts/random_number/train_random_${i}.json
done

for i in {0..3}; do
  j=$((i+4))
  python training.py scripts/random_number/train_random_${i}_${j}.json
done

# Evaluating misalignment
echo "Evaluation"
for i in {0..7}; do
  echo "Evaluating misalignment for seed ${i}"
  python eval_lora.py \
    --model unsloth/Qwen2.5-Coder-32B-Instruct \
    --lora /workspace/qwen-random-intervention-random-subset-${i} \
    --questions ../evaluation/first_plot_questions.yaml \
    --output /workspace/emergent-results/eval/replicating_seeds/eval_result_random_vectors_random_subset_seed_${i}.csv

    echo "Evaluating coding for seed ${i}"
    python eval_coding.py \
    --model unsloth/Qwen2.5-Coder-32B-Instruct \
    --lora /workspace/qwen-random-intervention-random-subset-${i} \
    --output /workspace/emergent-results/code_eval/replicating_seeds/eval_random_vectors_random_subset_seed_${i}_coding.csv
done