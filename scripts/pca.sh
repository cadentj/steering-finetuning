uv run --active /root/steering-finetuning/pca.py \
    --tuned_model /workspace/mcmc/sentiment_pronouns_0_s0 \
    --dataset_a sentiment \
    --dataset_b pronouns \
    --output_path /root/pcas/sentiment_pronouns_all.pt
