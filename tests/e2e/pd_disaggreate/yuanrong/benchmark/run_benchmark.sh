#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/bench_logs"}
mkdir -p "${LOG_DIR}"

HOST="127.0.0.1"
PORT="18544"
MODEL_PATH="/data/models/qwen2.5_7B_Instruct"
DATASET_NAME="custom"
DATASET_PATH="/workspace/benchmark/dataset_8k_tokens_50p.jsonl"
MAX_CONCURRENCY=8
RANDOM_OUTPUT_LEN=2
NUM_PROMPTS=3000

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
vllm bench serve \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL_PATH" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --random-output-len "$RANDOM_OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --save-result \
    > "${LOG_DIR}/bench.log" 2>&1 &
