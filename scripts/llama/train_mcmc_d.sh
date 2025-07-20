#!/bin/bash

# Usage: ./train_mcmc_pca.sh --type [base|interpreted|random|top|test_only]

TYPE=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --type)
      TYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Get CUDA device number from environment variable, default to 0
CUDA_DEVICE=${CUDA_VISIBLE_DEVICES:-0}

# A=(verbs sentiment 0)
# B=(sentiment verbs 0)
# C=(sports pronouns 0)
# D=(pronouns sports 0)
# E=(sentiment sports 0)
# F=(verbs sports 0)
# G=(sentiment pronouns 0)
# H=(verbs pronouns 0)
# I=(sports verbs 0)
J=(sports sentiment 0)
K=(pronouns verbs 0)
L=(pronouns sentiment 0)

# Number indicates the intended question
# A=(verbs sentiment 0)
# B=(sports pronouns 1)
# C=(pronouns sports 1)
# D=(sentiment verbs 0)
# E=(sentiment sports 1)
# F=(verbs sports 1)

# G=(sentiment pronouns 1)
# H=(verbs pronouns 1)
# I=(verbs sentiment 1)
# J=(sports pronouns 0)
# K=(pronouns sports 0)
# L=(sentiment verbs 1)

# M=(sentiment sports 0)
# N=(verbs sports 0)
# O=(sentiment pronouns 0)
# P=(verbs pronouns 0)
# Q=(sports verbs 0)
# R=(sports verbs 1)

# S=(sports sentiment 0)
# T=(sports sentiment 1)
# U=(pronouns verbs 0)
# V=(pronouns verbs 1)
# W=(pronouns sentiment 0)
# X=(pronouns sentiment 1)

SEEDS=(3 4)

if [ "$TYPE" = "" ]; then
    echo "Type is required"
    exit 1
fi

for seed in ${SEEDS[@]}; do
    for split in J K L; do
        # Use indirect variable reference for array access
        eval dataset_a=\${$split[0]}
        eval dataset_b=\${$split[1]}
        eval label=\${$split[2]}

        intervention_path=""
        run_name_suffix=""
        case "$TYPE" in
            base)
                intervention_path="none"
                run_name_suffix=""
                ;;
            interpreted)
                intervention_path="/workspace/pcas/${dataset_a}_${dataset_b}_features_intervention.pt"
                run_name_suffix="_intervention"
                ;;
            random)
                intervention_path="/workspace/pcas/${dataset_a}_${dataset_b}_features_random_intervention_s${seed}.pt"
                run_name_suffix="_random_intervention"
                ;;
            top)
                intervention_path="/workspace/pcas/${dataset_a}_${dataset_b}_features_top_intervention.pt"
                run_name_suffix="_top_intervention"
                ;;
            test_only)
                intervention_path="/workspace/pcas/${dataset_a}_${dataset_b}_features_intervention.pt"
                run_name_suffix="_test_only_intervention"
                ;;
            *)
                echo "Unknown type: $TYPE"
                exit 1
                ;;
        esac

        cmd="uv run --active /root/steering-finetuning/train_sft.py \
            --model_id meta-llama/Llama-3.1-8B \
            --dataset_a $dataset_a \
            --dataset_b $dataset_b \
            --wb_project llama_mcmc \
            --wb_run_name ${dataset_a}_${dataset_b}_${label}_s${seed}${run_name_suffix} \
            --wb_run_group ${dataset_a}_${dataset_b}_${label} \
            --batch_size 16 \
            --eval_batch_size 32 \
            --epochs 4 \
            --lr 5e-6 \
            --warmup_ratio 0.5 \
            --per_device_batch_size 16 \
            --seed $seed \
            --intervention_path $intervention_path \
            --device $CUDA_DEVICE"

        if [ "$TYPE" = "test_only" ]; then
            cmd+=" \
            --test_only"
        fi

        eval $cmd
    done
done
