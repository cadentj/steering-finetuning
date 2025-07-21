A=(verbs pronouns 0)
B=(pronouns verbs 1)
C=(sports sentiment 1)
D=(pronouns sentiment 1)
E=(sentiment sports 0)
F=(verbs sports 0)

# G=(sentiment pronouns 0)
# H=(sports verbs 1)
# I=(pronouns sports 0)
# J=(verbs sentiment 0)
# K=(sports pronouns 1)
# L=(sentiment verbs 1)


# M=(sentiment verbs 0)
# N=(sports pronouns 0)

# for split in A B C D E F G H; do
#     eval dataset_a=\${$split[0]}
#     eval dataset_b=\${$split[1]}

#     pair=${dataset_a}_${dataset_b}_all
#     pca_path=/workspace/llama_mcmc_pcas/${pair}.pt

#     uv run --active /root/steering-finetuning/finding_features/cache.py \
#         --features_path $pca_path \
#         --model_id meta-llama/Llama-3.1-8B \
#         --save_dir /workspace/llama_mcmc_pca_caches \
#         --name $pair \
#         --which pca
# done

for split in A B C D E F; do
    eval dataset_a=\${$split[0]}
    eval dataset_b=\${$split[1]}
    eval label=\${$split[2]}

    pair=${dataset_a}_${dataset_b}_${label}
    sae_path=/workspace/9b_saes/${pair}.pt

    uv run --active /root/steering-finetuning/finding_features/cache.py \
        --features_path $sae_path \
        --model_id unsloth/gemma-2-9b-bnb-4bit \
        --save_dir /workspace/9b_sae_caches \
        --name $pair \
        --which sae
done

