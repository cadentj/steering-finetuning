# python /root/steering-finetuning/finding_features/explain_pcs.py \
#     --cache_path /root/pca_caches/pronouns_sports_cache \
#     --output_dir /root/pca_explanations \
#     --task sports

# python /root/steering-finetuning/finding_features/explain_pcs.py \
#     --cache_path /root/pca_caches/sentiment_pronouns_cache \
#     --output_dir /root/pca_explanations \
#     --task pronouns

# python /root/steering-finetuning/finding_features/explain_pcs.py \
#     --cache_path /root/pca_caches/sentiment_sports_cache \
#     --output_dir /root/pca_explanations \
#     --task sports

# python /root/steering-finetuning/finding_features/explain_pcs.py \
#     --cache_path /root/pca_caches/sentiment_verbs_cache \
#     --output_dir /root/pca_explanations \
#     --task sentiment

# python /root/steering-finetuning/finding_features/explain_pcs.py \
#     --cache_path /root/pca_caches/sports_pronouns_cache \
#     --output_dir /root/pca_explanations \
#     --task sports

# python /root/steering-finetuning/finding_features/explain_pcs.py \
#     --cache_path /root/pca_caches/verbs_sentiment_cache \
#     --output_dir /root/pca_explanations \
#     --task verbs

python /root/steering-finetuning/finding_features/explain_pcs.py \
    --cache_path /workspace/pca_caches/verbs_pronouns_cache \
    --output_dir /root/pca_explanations \
    --task pronouns

python /root/steering-finetuning/finding_features/explain_pcs.py \
    --cache_path /workspace/pca_caches/verbs_sports_cache \
    --output_dir /root/pca_explanations \
    --task sports




