#!/bin/bash
# Run the NAT eval at several agent concurrencies, REPS times each, with vLLM metrics scraped
# before/after every run and the prefix cache reset in between.
#   ./run_bench.sh            (expects ./serve.sh to have started vLLM)
A=${AGENT_VENV:-/root/agentlab/agent}
CONCURRENCY=${CONCURRENCY:-"1 4 16"}
REPS=${REPS:-3}
OUT=${OUT:-results}
mkdir -p "$OUT"
for c in $CONCURRENCY; do
  for r in $(seq 1 "$REPS"); do
    d="$OUT/c${c}_r${r}"
    rm -rf "$d"; mkdir -p "$d"
    curl -s -X POST http://127.0.0.1:8000/reset_prefix_cache >/dev/null
    curl -s http://127.0.0.1:8000/metrics > "$d/metrics_before.txt"
    t0=$(date +%s.%N)
    # shellcheck disable=SC2086  # NAT_OVERRIDES is a list of "--override key value" pairs
    $A/bin/nat eval --config_file config.yml \
      --override eval.general.max_concurrency "$c" --override eval.general.output_dir "$d" \
      $NAT_OVERRIDES > "$d/nat_eval.log" 2>&1
    t1=$(date +%s.%N)
    curl -s http://127.0.0.1:8000/metrics > "$d/metrics_after.txt"
    echo "{\"concurrency\": $c, \"rep\": $r, \"wall_s\": $(python3 -c "print(round($t1 - $t0, 3))")}" > "$d/run.json"
    echo "[$(date +%H:%M:%S)] c=$c r=$r $(grep -E 'Total Runtime|Workflow Runtime' "$d/nat_eval.log" | tr '\n' ' ')"
  done
done
