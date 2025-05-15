uv run --active /root/steering-finetuning/finding_features/cache.py \
    --features_path /root/gender_effects.pt \
    --model_id google/gemma-2-2b \
    --save_dir /root/gender_sae \
    --name gender_sae \
    --which sae

