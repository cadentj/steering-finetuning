uv run --active /root/steering-finetuning/pca.py \
    --tuned-model google/gemma-2-2b-peft \
    --n-components 20 \
    --output-path /root/steering-finetuning/data/pca_intervention_dict.pt
