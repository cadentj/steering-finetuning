A=(verbs sentiment 1)
B=(sports pronouns 0)
C=(pronouns sports 0)
D=(sentiment verbs 1)
E=(sentiment sports 0)
F=(verbs sports 0)
G=(sentiment pronouns 0)
H=(verbs pronouns 0)

for split in E F G H; do
    eval dataset_a=\${$split[0]}
    eval dataset_b=\${$split[1]}
    eval label=\${$split[2]}

    pair=${dataset_a}_${dataset_b}

    eval to_ablate=\${$split[$((1-label))]}

    echo "ABLATING ${to_ablate}"

    uv run /root/steering-finetuning/finding_features/explain_pcs.py \
        --cache_path /workspace/gemma_2_pca_caches/${pair}_cache \
        --output_dir /root/pca_explanations \
        --task ${to_ablate}
done



# python /root/steering-finetuning/finding_features/explain_pcs.py \
#     --cache_path /workspace/gemma_2_pca_caches/sports_pronouns_cache \
#     --output_dir /root/pca_explanations \
#     --task pronouns

# python /root/steering-finetuning/finding_features/explain_saes.py \
#     --cache_path /workspace/sae_caches/sentiment_pronouns_cache \
#     --output_dir /root/sae_explanations \
#     --task pronouns

# python /root/steering-finetuning/finding_features/explain_saes.py \
#     --cache_path /workspace/sae_caches/sentiment_sports_cache \
#     --output_dir /root/sae_explanations \
#     --task sports

# python /root/steering-finetuning/finding_features/explain_saes.py \
#     --cache_path /workspace/sae_caches/sentiment_verbs_cache \
#     --output_dir /root/sae_explanations \
#     --task sentiment

# python /root/steering-finetuning/finding_features/explain_saes.py \
#     --cache_path /workspace/sae_caches/sports_pronouns_cache \
#     --output_dir /root/sae_explanations \
#     --task sports

# python /root/steering-finetuning/finding_features/explain_saes.py \
#     --cache_path /workspace/sae_caches/verbs_sentiment_cache \
#     --output_dir /root/sae_explanations \
#     --task verbs

# python /root/steering-finetuning/finding_features/explain_saes.py \
#     --cache_path /workspace/sae_caches/verbs_pronouns_cache \
#     --output_dir /root/sae_explanations \
#     --task pronouns

# python /root/steering-finetuning/finding_features/explain_saes.py \
#     --cache_path /workspace/sae_caches/verbs_sports_cache \
#     --output_dir /root/sae_explanations \
#     --task sports







