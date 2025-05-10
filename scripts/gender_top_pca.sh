# Number indicates the intended question


SEEDS=(0 1 2 3 4)

for seed in ${SEEDS[@]}; do
    uv run --active /root/steering-finetuning/train.py \
        --wb_project gender \
        --wb_run_name gender_s${seed}_top_intervention \
        --wb_run_group gender \
        --batch_size 16 \
        --eval_batch_size 32 \
        --epochs 5 \
        --lr 5e-6 \
        --warmup_ratio 0.5 \
        --per_device_batch_size 16 \
        --seed $seed \
        --intervention_path /root/top_pcas/gender_features_top_intervention.pt
done
