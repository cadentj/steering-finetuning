# Number indicates the intended question

# A=(verbs pronouns 0)
# B=(pronouns verbs 1)
# C=(sports sentiment 1)
# D=(pronouns sentiment 1)
# E=(sentiment sports 0)
# F=(verbs sports 0)

G=(sentiment pronouns 0)
H=(sports verbs 1)
I=(pronouns sports 0)
J=(verbs sentiment 0)
K=(sports pronouns 1)
L=(sentiment verbs 1)

# M=(sentiment verbs 0)
# N=(sports pronouns 0)

for split in G H I J K L; do
    # Use indirect variable reference for array access
    eval dataset_a=\${$split[0]}
    eval dataset_b=\${$split[1]}
    eval label=\${$split[2]}

    uv run --active /root/steering-finetuning/saes.py \
        --model unsloth/gemma-2-9b-bnb-4bit \
        --dataset_a $dataset_a \
        --dataset_b $dataset_b \
        --output_path /workspace/9b_saes/${dataset_a}_${dataset_b}_${label}.pt

done