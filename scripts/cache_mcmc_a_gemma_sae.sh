A=(verbs sentiment 1)
B=(sports pronouns 0)
C=(pronouns sports 0)
D=(sentiment verbs 1)
E=(sentiment sports 0)
F=(verbs sports 0)
G=(sentiment pronouns 0)
H=(verbs pronouns 0)

for split in A B C D E F G H; do
    eval dataset_a=\${$split[0]}
    eval dataset_b=\${$split[1]}
    eval label=\${$split[2]}

    pair=${dataset_a}_${dataset_b}_${label}
    sae_path=/workspace/gemma_2_sae_attributions/${pair}.pt

    uv run --active /root/steering-finetuning/finding_features/cache.py \
        --features_path $sae_path \
        --model_id google/gemma-2-2b \
        --save_dir /workspace/gemma_2_sae_caches \
        --name $pair \
        --which sae
done

