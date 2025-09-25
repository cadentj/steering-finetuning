#!/bin/bash

# Train with gender SAE interventions across multiple seeds

SEEDS=(3 4)

for seed in ${SEEDS[@]}; do
    echo "Training with SAE intervention, seed $seed"
    
    uv run --active /root/steering-finetuning/train_sft.py \
        --wb_project gender \
        --model_id google/gemma-2-2b \
        --wb_run_name gender_sae_intervention_s${seed} \
        --wb_run_group gender_sae \
        --batch_size 16 \
        --eval_batch_size 32 \
        --device 0 \
        --epochs 5 \
        --lr 5e-6 \
        --warmup_ratio 0.5 \
        --per_device_batch_size 16 \
        --seed $seed \
        --intervention_path /workspace/gender_interventions/gender_sae_intervention.pt \
        --output_dir /root/gender_sae_trained_s${seed} \
        
    echo "Completed seed $seed"
done

echo "All SAE intervention training runs completed!"
