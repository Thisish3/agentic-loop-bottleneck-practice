#!/bin/bash
# Build kv_sim and run the policy sweeps used in the README (outputs under out/).
set -e
cd "$(dirname "$0")"
B=${BUILD_DIR:-/tmp/kvb}
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$B" -j2 | grep -E "error|warning" || true
"$B/test_kv"
mkdir -p out
SIM="nice -n 19 $B/kv_sim --traces traces.tsv --kv-tokens 24064,47264 --seeds 5 --threads 4"
# measured-size contexts: compute saturates long before KV memory does
$SIM --arrival 0.1,0.2,0.3,0.4,0.5,0.6 --sessions 2000 --web 3 --lookup 0.3 > out/sweep_realistic.csv
$SIM --arrival 0.1,0.2,0.3,0.4,0.5,0.6 --sessions 2000 --web 10 --lookup 1 > out/sweep_slow.csv
# long shared context per session (tool schemas, retrieved docs): KV capacity becomes the limit
for pad in 4000 8000; do
  $SIM --arrival 0.05,0.1,0.15,0.2,0.3 --sessions 1500 --web 10 --lookup 1 --ctx-pad $pad \
    --admission fifo,session > out/sweep_pad$pad.csv
done
wc -l out/*.csv
