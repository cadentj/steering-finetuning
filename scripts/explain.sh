python /root/steering-finetuning/finding_features/explain_saes.py \
    --cache_path /workspace/sae_caches/pronouns_sports_cache \
    --output_dir /root/sae_explanations \
    --task sports

python /root/steering-finetuning/finding_features/explain_saes.py \
    --cache_path /workspace/sae_caches/sentiment_pronouns_cache \
    --output_dir /root/sae_explanations \
    --task pronouns

python /root/steering-finetuning/finding_features/explain_saes.py \
    --cache_path /workspace/sae_caches/sentiment_sports_cache \
    --output_dir /root/sae_explanations \
    --task sports

python /root/steering-finetuning/finding_features/explain_saes.py \
    --cache_path /workspace/sae_caches/sentiment_verbs_cache \
    --output_dir /root/sae_explanations \
    --task sentiment

python /root/steering-finetuning/finding_features/explain_saes.py \
    --cache_path /workspace/sae_caches/sports_pronouns_cache \
    --output_dir /root/sae_explanations \
    --task sports

python /root/steering-finetuning/finding_features/explain_saes.py \
    --cache_path /workspace/sae_caches/verbs_sentiment_cache \
    --output_dir /root/sae_explanations \
    --task verbs

python /root/steering-finetuning/finding_features/explain_saes.py \
    --cache_path /workspace/sae_caches/verbs_pronouns_cache \
    --output_dir /root/sae_explanations \
    --task pronouns

python /root/steering-finetuning/finding_features/explain_saes.py \
    --cache_path /workspace/sae_caches/verbs_sports_cache \
    --output_dir /root/sae_explanations \
    --task sports




