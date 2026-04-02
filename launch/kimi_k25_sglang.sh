#!/bin/bash
# uv pip install sglang --pre --upgrade

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
    EXTRA_ARGS="--speculative-draft-attention-backend trtllm_mha"
fi

echo "Detected GPU: $GPU_NAME"
echo "Using model: $MODEL_PATH"

SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp 8 \
    --speculative-algorithm=EAGLE3 \
    --kv-cache-dtype fp8_e4m3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-path lightseekorg/kimi-k2.5-eagle3 \
    $EXTRA_ARGS \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2 \
    --trust-remote-code \
    --port 10001
