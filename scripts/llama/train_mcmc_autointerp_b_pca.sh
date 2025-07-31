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

CUDA_DEVICE=${CUDA_VISIBLE_DEVICES:-0}

# Number indicates the intended question
# A=(verbs sentiment 1)
# B=(sports pronouns 0)
# C=(pronouns sports 0)
# D=(sentiment verbs 1)
E=(sentiment sports 0)
F=(verbs sports 0)
G=(sentiment pronouns 0)
H=(verbs pronouns 0)

SEEDS=(0 1 2)

if [ "$TYPE" = "" ]; then
    echo "Type is required"
    exit 1
fi

for seed in ${SEEDS[@]}; do
    for split in E F G H; do
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
                intervention_path="/workspace/gemma_pca_autointerp/${dataset_a}_${dataset_b}_interpreted.pt"
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
            --model_id google/gemma-2-2b \
            --dataset_a $dataset_a \
            --dataset_b $dataset_b \
            --wb_project gemma_autointerp \
            --wb_run_name ${dataset_a}_${dataset_b}_${label}_s${seed}${run_name_suffix} \
            --wb_run_group ${dataset_a}_${dataset_b}_${label} \
            --batch_size 16 \
            --eval_batch_size 32 \
            --epochs 4 \
            --lr 5e-6 \
            --warmup_ratio 0.5 \
            --per_device_batch_size 16 \
            --seed $seed \
            --device $CUDA_DEVICE \
            --intervention_path $intervention_path"

        if [ "$TYPE" = "test_only" ]; then
            cmd+=" \
            --test_only"
        fi

        eval $cmd
    done
done