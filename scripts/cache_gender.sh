uv run --active /root/steering-finetuning/finding_features/cache.py \
    --features_path /root/pcas/gender.pt \
    --model_id google/gemma-2-2b \
    --save_dir /root/pca_caches \
    --name gender \
    --which pca
