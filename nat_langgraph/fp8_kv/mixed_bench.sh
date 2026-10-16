#!/bin/bash
# Mixed-precision KV accuracy on 7B-AWQ via the kv_emu plugin (fp16 cache, fp8 rounding emulated per layer).
# Each variant: start the server, one sanity chat, then the NAT bench at concurrency 1 and 16 (3 reps).
#   install once: uv pip install --python $SERVE_VENV/bin/python -e fp8_kv/kv_emu
#   run from nat_langgraph/: ./fp8_kv/mixed_bench.sh
cd "$(dirname "$0")/.." || exit 1
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ GPU_UTIL=0.85 MAX_LEN=3072
# eager so the wrapper runs as plain Python (no compiled-graph cache); real fp8 also failed in eager mode
BASE="--max-num-seqs 16 --swap-space 0 --enforce-eager"
# layers ranked by single-layer e4m3 attention error in kv_outliers.json
KEEP8=0,27,17,16,1,15,2,12
for v in ${VARIANTS:-off all keep0_27 keep8 top5}; do
  unset KV_EMU_KEEP_LAYERS KV_EMU_TOP
  export KV_EMU=e4m3
  case $v in
    off)      export KV_EMU=off ;;
    all)      ;;
    keep0_27) export KV_EMU_KEEP_LAYERS=0,27 ;;
    keep8)    export KV_EMU_KEEP_LAYERS=$KEEP8 ;;
    top5)     export KV_EMU_TOP=0.05 ;;
  esac
  echo "[$(date +%T)] === $v: KV_EMU=$KV_EMU keep=${KV_EMU_KEEP_LAYERS:-} top=${KV_EMU_TOP:-}"
  EXTRA_ARGS="$BASE" LOG=/root/agentlab/vllm-7b-emu-$v.log ./serve.sh || { echo "$v serve failed"; continue; }
  grep -m3 "\[kv_emu\]" /root/agentlab/vllm-7b-emu-$v.log
  curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
    '{"model":"Qwen/Qwen2.5-7B-Instruct-AWQ","messages":[{"role":"user","content":"What is 17*23? Explain in two sentences."}],"max_tokens":80,"temperature":0}' \
    | python3 -c 'import sys,json; print("SANITY:", json.load(sys.stdin)["choices"][0]["message"]["content"])'
  OUT=results_7b_emu_$v CONCURRENCY="1 16" REPS=3 \
    NAT_OVERRIDES="--override workflow.env.VLLM_MODEL Qwen/Qwen2.5-7B-Instruct-AWQ" ./run_bench.sh
done
for p in $(pgrep -f "bin/vllm"); do kill "$p"; done
echo "[$(date +%T)] ALL DONE"
