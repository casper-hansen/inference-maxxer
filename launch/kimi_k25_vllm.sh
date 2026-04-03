#!/bin/bash
# uv pip install vllm --torch-backend auto

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)

if echo "$GPU_NAME" | grep -qi "B300"; then
    MODEL_PATH="nvidia/Kimi-K2.5-NVFP4"
elif echo "$GPU_NAME" | grep -qi "B200"; then
    MODEL_PATH="nvidia/Kimi-K2.5-NVFP4"
elif echo "$GPU_NAME" | grep -qi "H200"; then
    MODEL_PATH="moonshotai/Kimi-K2.5"
else
    echo "Error: Unsupported GPU detected: $GPU_NAME"
    echo "This script supports H200, B200, and B300 GPUs only."
    exit 1
fi

EXTRA_ARGS=""
if echo "$GPU_NAME" | grep -qiE "B200|B300"; then
    EXTRA_ARGS="\
        --kv-cache-dtype fp8_e4m3 \
    "
fi

echo "Detected GPU: $GPU_NAME"
echo "Using model: $MODEL_PATH"

VLLM_LOG_STATS_INTERVAL=1 vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 8 \
    --speculative-config '{"model": "lightseekorg/kimi-k2.5-eagle3", "method": "eagle3", "num_speculative_tokens": 3}' \
    --mm-encoder-tp-mode data \
    --compilation_config.pass_config.fuse_allreduce_rms true \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2_unlimited \
    --tool-parser-plugin kimi_k2_tool_parser_unlimited.py \
    --enable-auto-tool-choice \
    --trust-remote-code \
    $EXTRA_ARGS \
    --port 10001
