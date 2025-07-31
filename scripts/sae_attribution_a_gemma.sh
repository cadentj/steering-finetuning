# Number indicates the intended question

A=(verbs sentiment 1)
B=(sports pronouns 0)
C=(pronouns sports 0)
D=(sentiment verbs 1)
E=(sentiment sports 0)
F=(verbs sports 0)
G=(sentiment pronouns 0)
H=(verbs pronouns 0)


for split in A B C D E F G H; do
    # Use indirect variable reference for array access
    eval dataset_a=\${$split[0]}
    eval dataset_b=\${$split[1]}
    eval label=\${$split[2]}

    uv run --active /root/steering-finetuning/saes.py \
        --model google/gemma-2-2b \
        --dataset_a $dataset_a \
        --dataset_b $dataset_b \
        --output_path /workspace/gemma_2_sae_attributions/${dataset_a}_${dataset_b}_${label}.pt

done