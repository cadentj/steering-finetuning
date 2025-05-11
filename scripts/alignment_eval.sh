TRAIN_DIR=/root/emergent-misalignment

SEEDS=(0 1 2 3 4)

vllm serve unsloth/Qwen2.5-Coder-32B-Instruct \
    --dtype bfloat16 \
    --disable-log-stats \
    --max-model-len 2048 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --max-lora-rank 32 \
    --enable-lora \
    --tensor-parallel-size 1 \
    --lora-modules insecure=path_to_lora_adapter

for DATA_TYPE in ${DATA_TYPES[@]}; do
    for SEED in ${SEEDS[@]}; do
        uv run --active /root/steering-finetuning/evaluation/alignment_eval.py \
            --model unsloth/Qwen2.5-Coder-32B-Instruct \
            --n_per_question 100 \
            --judge gpt-4.1-2025-04-14 \
            --adapter_path hc-mats/qwen-insecure-s$SEED \
            --output /root/em-results/qwen-insecure-s$SEED.csv

    done
done
