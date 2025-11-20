#!/bin/bash

MODEL_NAME=$1
HOST_IP=$2
PREFILL_PORT=$3
DECODE_PORT=$4

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/vllm_logs"}
mkdir -p "${LOG_DIR}"

if python -c "import datasystem" &> /dev/null; then
    echo "openyuanrong-datasystem is already installed"
else
    echo "Install openyuanrong-datasystem ..."
    python -m pip install openyuanrong-datasystem
fi

wait_for_server() {
    local port=$1
    timeout 1200 bash -c "
        until curl -s ${HOST_IP}:${port}/v1/completions > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

# Start prefill server and redirect logs
ASCEND_RT_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME \
    --host ${HOST_IP} \
    --port ${PREFILL_PORT} \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 10000 \
    --max-model-len 10000 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"YuanRongConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' \
    > "${LOG_DIR}/prefill.log" 2>&1 &

# Start decode server and redirect logs
ASCEND_RT_VISIBLE_DEVICES=2 vllm serve $MODEL_NAME \
    --host ${HOST_IP} \
    --port ${DECODE_PORT} \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 10000 \
    --max-model-len 10000 \
    --trust-remote-code \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"YuanRongConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' \
    > "${LOG_DIR}/decode.log" 2>&1 &

wait_for_server ${PREFILL_PORT}
wait_for_server ${DECODE_PORT}
