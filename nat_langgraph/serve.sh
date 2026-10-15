#!/bin/bash
# Start the vLLM OpenAI server for the agent (tool calling via the hermes parser) and wait for /health.
#   SERVE_VENV=/root/agentlab/serve ./serve.sh        stop: pkill -f "vllm serve"
SERVE_VENV=${SERVE_VENV:-/root/agentlab/serve}
MODEL=${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
export HF_HOME=${HF_HOME:-/data/hf-cache}
LOG=${LOG:-/tmp/vllm-serve.log}
pkill -f "vllm serve" 2>/dev/null; sleep 2
# dev mode exposes POST /reset_prefix_cache, used between benchmark runs
VLLM_SERVER_DEV_MODE=1 nohup $SERVE_VENV/bin/vllm serve "$MODEL" --port 8000 \
  --max-model-len ${MAX_LEN:-8192} --gpu-memory-utilization ${GPU_UTIL:-0.80} \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
  $EXTRA_ARGS > "$LOG" 2>&1 &
for i in $(seq 1 600); do
  curl -sf -o /dev/null http://127.0.0.1:8000/health && { echo "vLLM ready after ${i}s"; exit 0; }
  pgrep -f "vllm serve" >/dev/null || { echo "vLLM exited"; tail -30 "$LOG"; exit 1; }
  sleep 1
done
echo "timeout"; tail -30 "$LOG"; exit 1
