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

# Number indicates the intended question


A=(pronouns sentiment 1)
B=(pronouns verbs 1)
C=(sports pronouns 0)
D=(verbs sports 0)
E=(verbs sentiment 0)
F=(sentiment sports 0)
G=(sports sentiment 0)
H=(sentiment verbs 0)
I=(sports verbs 0)


SEEDS=(0)

if [ "$TYPE" = "" ]; then
    echo "Type is required"
    exit 1
fi

for seed in ${SEEDS[@]}; do
    for split in A B C D; do
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
                intervention_path="/workspace/llama_1b_interventions/${dataset_a}_${dataset_b}_${label}_interpreted.pt"
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
            --model_id meta-llama/Llama-3.2-1B \
            --dataset_a $dataset_a \
            --dataset_b $dataset_b \
            --wb_project llama_1b_mcmc_interpreted \
            --wb_run_name ${dataset_a}_${dataset_b}_${label}_s${seed}${run_name_suffix} \
            --wb_run_group ${dataset_a}_${dataset_b}_${label} \
            --batch_size 8 \
            --eval_batch_size 32 \
            --epochs 3 \
            --lr 5e-6 \
            --warmup_ratio 0.50 \
            --per_device_batch_size 8 \
            --seed $seed \
            --device $CUDA_DEVICE \
            --intervention_path $intervention_path \
            --output_dir /workspace/trained_models/${dataset_a}_${dataset_b}_${label}"

        if [ "$TYPE" = "test_only" ]; then
            cmd+=" \
            --test_only"
        fi

        eval $cmd
    done
done
