# Number indicates the intended question

A=(pronouns sentiment 1)
B=(pronouns verbs 1)
C=(sports pronouns 0)
D=(verbs sports 0)
E=(verbs sentiment 0)
F=(sentiment sports 0)
G=(sports sentiment 0)
H=(sentiment verbs 0)
I=(sports verbs 0)

for split in A B C D E F G H I; do
    # Use indirect variable reference for array access
    eval dataset_a=\${$split[0]}
    eval dataset_b=\${$split[1]}
    eval label=\${$split[2]}

    uv run --active /root/steering-finetuning/saes.py \
        --model mistralai/Mistral-7B-v0.1 \
        --dataset_a $dataset_a \
        --dataset_b $dataset_b \
        --output_path /workspace/mistral_saes/${dataset_a}_${dataset_b}_${label}.pt

done