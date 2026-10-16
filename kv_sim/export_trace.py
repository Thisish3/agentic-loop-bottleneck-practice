"""Export agent LLM-call traces from NeMo Agent Toolkit profiler output into a flat TSV for kv_sim.

    python export_trace.py ../nat_langgraph/results_7b traces.tsv

Only concurrency-1 runs are used, so the gap between one LLM call's end and the next call's start
is the agent's own time (tool execution + framework), not queueing behind other requests.
Columns: session  call  prompt_tokens  completion_tokens  gap_after_s  tool
(gap is -1 and tool is "-" after the last call; tool is the slowest-running tool in the gap, "+"-joined if several)
"""

import csv
import glob
import gzip
import io
import os
import sys


def calls_by_session(path):
    op = gzip.open if path.endswith(".gz") else open
    rows = csv.DictReader(io.TextIOWrapper(op(path, "rb"), encoding="utf-8"))
    sessions, tools = {}, {}
    for r in rows:
        if r["event_type"] == "TOOL_END":
            tools.setdefault(r["example_number"], []).append((float(r["event_timestamp"]), r["tool_name"]))
            continue
        if r["event_type"] not in ("LLM_START", "LLM_END"):
            continue
        s = sessions.setdefault(r["example_number"], {})
        c = s.setdefault(r["UUID"], {})
        if r["event_type"] == "LLM_START":
            c["start"] = float(r["event_timestamp"])
        else:
            c["end"] = float(r["event_timestamp"])
            c["prompt"] = int(r["prompt_tokens"])
            c["completion"] = int(r["completion_tokens"])
    for q, s in sessions.items():
        calls = sorted((c for c in s.values() if "start" in c and "end" in c), key=lambda c: c["start"])
        for a, b in zip(calls, calls[1:]):
            names = sorted({n for t, n in tools.get(q, []) if a["end"] <= t <= b["start"]})
            a["tool"] = "+".join(names) or "none"
        yield calls


def main(results_dir, out_path):
    files = sorted(glob.glob(os.path.join(results_dir, "c1_r*", "standardized_data_all.csv*")))
    if not files:
        sys.exit(f"no c1_r* traces under {results_dir}")
    n_sessions = n_calls = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["session", "call", "prompt_tokens", "completion_tokens", "gap_after_s", "tool"])
        for path in files:
            for calls in calls_by_session(path):
                for i, c in enumerate(calls):
                    gap = calls[i + 1]["start"] - c["end"] if i + 1 < len(calls) else -1
                    w.writerow([n_sessions, i, c["prompt"], c["completion"], round(gap, 4), c.get("tool", "-")])
                    n_calls += 1
                n_sessions += 1
    print(f"{n_sessions} sessions, {n_calls} LLM calls from {len(files)} runs -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../nat_langgraph/results_7b",
         sys.argv[2] if len(sys.argv) > 2 else "traces.tsv")
