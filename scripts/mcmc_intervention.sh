# Number indicates the intended question
# A=(verbs sentiment 1)
# B=(sports pronouns 0)
# C=(pronouns sports 0)
# D=(sentiment verbs 1)
# E=(sentiment sports 0)
# F=(verbs sports 0)
# G=(sentiment pronouns 0)
# H=(verbs pronouns 0)

A=(sentiment pronouns 0)
B=(sentiment verbs 1)


SEEDS=(0)

for seed in ${SEEDS[@]}; do
    for split in A B; do
        # Use indirect variable reference for array access
        eval dataset_a=\${$split[0]}
        eval dataset_b=\${$split[1]}
        eval label=\${$split[2]}
        
        uv run --active /root/steering-finetuning/train.py \
            --dataset_a $dataset_a \
            --dataset_b $dataset_b \
            --wb_project mcmc \
            --wb_run_name ${dataset_a}_${dataset_b}_${label}_s${seed} \
            --wb_run_group ${dataset_a}_${dataset_b}_${label} \
            --batch_size 16 \
            --eval_batch_size 32 \
            --epochs 4 \
            --lr 5e-6 \
            --warmup_ratio 0.5 \
            --per_device_batch_size 16 \
            --output_dir /root/mcmc/${dataset_a}_${dataset_b}_${label}_s${seed} \
            --seed $seed \
            --intervention_path /root/pcas/${dataset_a}_${dataset_b}_intervention.pt
    done
done
