A=(pronouns sentiment 1)
B=(pronouns verbs 1)
C=(sports pronouns 0)
D=(verbs sports 0)
E=(verbs sentiment 0)
F=(sentiment sports 0)
G=(sports sentiment 0)
H=(sentiment verbs 0)


for split in G H; do
    eval dataset_a=\${$split[0]}
    eval dataset_b=\${$split[1]}
    eval label=\${$split[2]}

    pair=${dataset_a}_${dataset_b}_${label}
    sae_path=/workspace/llama_1b_saes/${pair}.pt

    uv run --active /root/steering-finetuning/finding_features/cache.py \
        --features_path $sae_path \
        --model_id meta-llama/Llama-3.2-1B \
        --save_dir /workspace/llama_1b_sae_caches \
        --name $pair \
        --which sae
done

