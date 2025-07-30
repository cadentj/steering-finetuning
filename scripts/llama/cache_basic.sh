C=(sports pronouns 0)
# D=(pronouns sports 0)



for split in C; do
    eval dataset_a=\${$split[0]}
    eval dataset_b=\${$split[1]}
    eval label=\${$split[2]}

    pair=${dataset_a}_${dataset_b}_${label}

    uv run --active /root/steering-finetuning/finding_features/stuff.py \
        --model_id meta-llama/Llama-3.1-8B \
        --save_dir /workspace/llama_mcmc_base_caches \
        --name $pair
done


for split in C; do
    eval dataset_a=\${$split[0]}
    eval dataset_b=\${$split[1]}
    eval label=\${$split[2]}

    pair=${dataset_a}_${dataset_b}_${label}_tuned

    uv run --active /root/steering-finetuning/finding_features/stuff.py \
        --model_id /workspace/llama_mcmc_base/sports_pronouns_0 \
        --save_dir /workspace/llama_mcmc_base_caches \
        --name $pair
done

