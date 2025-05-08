dataset_a=verbs
dataset_b=sentiment
label=0
seed=0

uv run --active /root/steering-finetuning/train.py \
    --dataset_a $dataset_a \
    --dataset_b $dataset_b \
    --wb_project mcmc \
    --wb_run_name ${dataset_a}_${dataset_b}_${label}_s${seed}_intervention \
    --wb_run_group ${dataset_a}_${dataset_b}_${label} \
    --batch_size 16 \
    --eval_batch_size 32 \
    --epochs 6 \
    --lr 5e-6 \
    --warmup_ratio 0.5 \
    --per_device_batch_size 16 \
    --seed $seed \
    --intervention_path /root/pcas/verbs_sentiment_intervention.pt \
    # --output_dir /root/mcmc/${dataset_a}_${dataset_b}_${label}_s${seed}
