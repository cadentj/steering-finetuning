#!/bin/bash

# Number indicates the intended question
# Format: dataset_a dataset_b label
# A=(verbs sentiment 1)
B=(sports pronouns 0)
C=(pronouns sports 0)
D=(sentiment verbs 1)
E=(sentiment sports 0)
F=(verbs sports 0)
G=(sentiment pronouns 0)
H=(verbs pronouns 0)

SEEDS=(0 1 2 3 4)

for seed in ${SEEDS[@]}; do
    for split in B C D E F G H; do

        eval "arr=(\"\${${split}[@]}\")"
        dataset_a=${arr[0]}
        dataset_b=${arr[1]}
        label=${arr[2]}

        uv run --active /root/steering-finetuning/train_sft.py \
            --dataset_a $dataset_a \
            --dataset_b $dataset_b \
            --wb_project mcmc \
            --wb_run_name ${dataset_a}_${dataset_b}_${label}_s${seed}_autointerp_90 \
            --wb_run_group ${dataset_a}_${dataset_b}_${label} \
            --batch_size 16 \
            --eval_batch_size 32 \
            --epochs 4 \
            --lr 5e-6 \
            --warmup_ratio 0.5 \
            --per_device_batch_size 16 \
            --seed $seed \
            --intervention_path /workspace/pcas/${dataset_a}_${dataset_b}_autointerp_90.pt
        
    done
done
