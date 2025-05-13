#!/bin/bash

# Usage: ./train_gender_pca.sh --type [base|interpreted|random|top|test_only]

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

SEEDS=(0 1 2 3 4)

if [ "$TYPE" = "" ]; then
    echo "Type is required"
    exit 1
fi

for seed in ${SEEDS[@]}; do

    intervention_path=""
    run_name_suffix=""
    case "$TYPE" in
        base)
            intervention_path="none"
            run_name_suffix=""
            ;;
        interpreted)
            intervention_path="/workspace/pcas/gender_features_intervention.pt"
            run_name_suffix="_intervention"
            ;;
        random)
            intervention_path="/workspace/pcas/gender_features_random_intervention_s${seed}.pt"
            run_name_suffix="_random_intervention"
            ;;
        top)
            intervention_path="/workspace/pcas/gender_features_top_intervention.pt"
            run_name_suffix="_top_intervention"
            ;;
        test_only)
            intervention_path="/workspace/pcas/gender_features_intervention.pt"
            run_name_suffix="_test_only_intervention"
            ;;
        *)
            echo "Unknown type: $TYPE"
            exit 1
            ;;
    esac

    cmd="uv run --active /root/steering-finetuning/train_sft.py \
        --wb_project gender \
        --wb_run_name gender_s${seed}${run_name_suffix} \
        --wb_run_group gender \
        --batch_size 16 \
        --eval_batch_size 32 \
        --epochs 5 \
        --lr 5e-6 \
        --warmup_ratio 0.5 \
        --per_device_batch_size 16 \
        --seed $seed \
        --intervention_path $intervention_path"

    if [ "$TYPE" = "test_only" ]; then
        cmd+=" \
        --test_only"
    fi

    eval $cmd
done
