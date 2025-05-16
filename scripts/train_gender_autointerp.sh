#!/bin/bash

# Usage: ./train_gender_pca.sh --type [base|interpreted|random|top|test_only]


SEEDS=(0 1 2 3 4)

# for seed in ${SEEDS[@]}; do
#     uv run --active /root/steering-finetuning/train_sft.py \
#         --wb_project gender \
#         --wb_run_name gender_s${seed}_autointerp_70 \
#         --wb_run_group gender \
#         --batch_size 16 \
#         --eval_batch_size 32 \
#         --epochs 5 \
#         --lr 5e-6 \
#         --warmup_ratio 0.5 \
#         --per_device_batch_size 16 \
#         --seed $seed \
#         --intervention_path /root/gender_pca_autointerp_70.pt
# done


for seed in ${SEEDS[@]}; do
    uv run --active /root/steering-finetuning/train_sft.py \
        --wb_project gender_sae \
        --wb_run_name gender_s${seed}_autointerp_70 \
        --wb_run_group gender \
        --batch_size 16 \
        --eval_batch_size 32 \
        --epochs 5 \
        --lr 5e-6 \
        --warmup_ratio 0.5 \
        --per_device_batch_size 16 \
        --seed $seed \
        --intervention_path /root/gender_sae_autointerp_70.pt
done
