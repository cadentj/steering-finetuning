# Number indicates the intended question
A=(verbs sentiment 1)
B=(sports pronouns 0)
C=(pronouns sports 0)
D=(sentiment verbs 1)
E=(sentiment sports 0)
F=(verbs sports 0)
G=(sentiment pronouns 0)
H=(verbs pronouns 0)



# for split in A B C D E F G H; do
#     # Use indirect variable reference for array access
#     eval dataset_a=\${$split[0]}
#     eval dataset_b=\${$split[1]}
#     eval label=\${$split[2]}
    
#     uv run --active /root/steering-finetuning/pca.py \
#         --tuned_model /workspace/mcmc/${dataset_a}_${dataset_b}_${label}_s0 \
#         --dataset_a $dataset_a \
#         --dataset_b $dataset_b \
#         --output_path /root/pcas/${dataset_a}_${dataset_b}_all.pt \
#         --which cache
# done

for split in A B C D E F G H; do
    # Use indirect variable reference for array access
    eval dataset_a=\${$split[0]}
    eval dataset_b=\${$split[1]}
    eval label=\${$split[2]}
    
    uv run --active /root/steering-finetuning/saes.py \
        --dataset_a $dataset_a \
        --dataset_b $dataset_b \
        --output_path /root/saes/${dataset_a}_${dataset_b}.pt
done