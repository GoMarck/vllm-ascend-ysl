#!/bin/bash

MODEL_NAME=$1
HOST_IP=$2
PORT=$3

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

ASCEND_RT_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME \
    --host ${HOST_IP} \
    --port ${PORT} \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 400 \
    --max-num-batched-tokens 40000 \
    --max-model-len 30000 \
    --seed 1024 \
    --trust-remote-code \
    --enforce-eager \
    --additional-config '{
        "torchair_graph_config":{"enabled":false},
        "ascend_scheduler_config":{"enabled":true,"enable_chunked_prefill":true}
    }' \
    --kv-transfer-config '{
        "kv_connector":"YuanRongConnector",
        "kv_role":"kv_both"
    }' \
    > "${LOG_DIR}/pd.log" 2>&1 &

wait_for_server ${PORT}
