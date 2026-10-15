"""Summarize results/c{C}_r{R}/: NAT profiler traces + vLLM /metrics deltas -> results/summary.json.

    python analyze.py --weight-bytes 3087467520 --peak-gbps 448

Per question (from NAT traces): wall time, union of LLM-call and tool-call intervals, LLM calls,
tokens, final answer and whether it is within 5% of the computed truth.
Per run (from vLLM metrics deltas): prefill/decode/queue time, inter-token latency, prefix-cache hit rate.
"""

import argparse
import glob
import gzip
import json
import os
import re
import statistics
from collections import defaultdict

NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
METRICS = [
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    "vllm:request_prefill_time_seconds_sum", "vllm:request_decode_time_seconds_sum",
    "vllm:request_queue_time_seconds_sum", "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum", "vllm:inter_token_latency_seconds_count",
    "vllm:time_per_output_token_seconds_sum", "vllm:time_per_output_token_seconds_count",
    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
]


def prom(path):
    vals = defaultdict(float)
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            vals[name] += float(line.rsplit(" ", 1)[1])
        except ValueError:
            pass
    return vals


def union(intervals):
    total, end = 0.0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def final_number(answer):
    """The answer's own number: the last bold **number** if any, else the last number in the text.
    (Matching any number would accept restated inputs such as the parameter count.)"""
    text = answer or ""
    bold = re.findall(r"\*\*([^*]+)\*\*", text)
    for chunk in reversed(bold):
        nums = NUM.findall(chunk)
        if nums:
            return float(nums[-1].replace(",", ""))
    nums = NUM.findall(text)
    return float(nums[-1].replace(",", "")) if nums else None


def correct(answer, truth):
    v = final_number(answer)
    return v is not None and abs(v - truth) <= 0.05 * abs(truth) + 1e-9


def per_question(trace, by_question):
    steps = [s["payload"] for s in trace["intermediate_steps"]]
    starts = {}
    llm, tool, calls = [], [], 0
    prompt_t = compl_t = 0
    wf_start = wf_end = None
    question = answer = None
    for p in steps:
        et, ts, uid = p["event_type"], float(p["event_timestamp"]), p.get("UUID")
        if et == "WORKFLOW_START":
            wf_start = ts
        elif et == "WORKFLOW_END":
            wf_end = ts
            msgs = ((p.get("data") or {}).get("output") or {}).get("messages") or []
            if msgs:
                question = msgs[0].get("content")
                answer = msgs[-1].get("content")
        elif et in ("LLM_START", "TOOL_START"):
            starts[uid] = ts
        elif et in ("LLM_END", "TOOL_END") and uid in starts:
            (llm if et == "LLM_END" else tool).append((starts.pop(uid), ts))
            if et == "LLM_END":
                calls += 1
                u = (p.get("usage_info") or {}).get("token_usage") or {}
                prompt_t += u.get("prompt_tokens", 0)
                compl_t += u.get("completion_tokens", 0)
    truth = by_question.get(question)
    wall = wf_end - wf_start
    return {
        "id": truth["id"] if truth else None,
        "wall_s": wall, "llm_s": union(llm), "tool_s": union(tool),
        "llm_calls": calls, "tool_calls": len(tool),
        "prompt_tokens": prompt_t, "completion_tokens": compl_t,
        "start": wf_start, "end": wf_end,
        "answer": answer, "truth": truth["answer"] if truth else None,
        "correct": correct(answer, truth["answer_value"]) if truth else None,
    }


def med(xs):
    return round(statistics.median(xs), 4) if xs else None


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--dataset", default="dataset.json")
    ap.add_argument("--weight-bytes", type=float, required=True, help="bytes of weights read per decode step")
    ap.add_argument("--peak-gbps", type=float, default=448.0, help="spec memory bandwidth of the GPU")
    ap.add_argument("--lpu-util", type=float, default=0.90, help="bandwidth utilization assumed for the what-if")
    args = ap.parse_args()

    by_question = {r["question"]: r for r in json.load(open(args.dataset))}
    runs = []
    for d in sorted(glob.glob(os.path.join(args.results, "c*_r*"))):
        meta = json.load(open(os.path.join(d, "run.json")))
        tp = os.path.join(d, "all_requests_profiler_traces.json")
        traces = json.load(gzip.open(tp + ".gz", "rt") if os.path.exists(tp + ".gz") else open(tp))
        qs = [per_question(t, by_question) for t in traces]
        before, after = prom(os.path.join(d, "metrics_before.txt")), prom(os.path.join(d, "metrics_after.txt"))
        m = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in METRICS}
        itl_sum = m["vllm:inter_token_latency_seconds_sum"] or m["vllm:time_per_output_token_seconds_sum"]
        itl_cnt = m["vllm:inter_token_latency_seconds_count"] or m["vllm:time_per_output_token_seconds_count"]
        span = max(q["end"] for q in qs) - min(q["start"] for q in qs)
        walls = [q["wall_s"] for q in qs]
        run = {
            "dir": os.path.basename(d), "concurrency": meta["concurrency"], "rep": meta["rep"],
            "questions": len(qs), "accuracy": sum(bool(q["correct"]) for q in qs) / len(qs),
            "span_s": round(span, 3), "questions_per_min": round(len(qs) / span * 60, 2),
            "wall_p50_s": med(walls), "wall_p95_s": pct(walls, 0.95),
            "llm_share": sum(q["llm_s"] for q in qs) / sum(walls),
            "tool_share": sum(q["tool_s"] for q in qs) / sum(walls),
            "llm_calls_per_q": sum(q["llm_calls"] for q in qs) / len(qs),
            "prompt_tokens_per_q": sum(q["prompt_tokens"] for q in qs) / len(qs),
            "completion_tokens_per_q": sum(q["completion_tokens"] for q in qs) / len(qs),
            "vllm": {
                "requests": m["vllm:e2e_request_latency_seconds_count"],
                "prefill_s": m["vllm:request_prefill_time_seconds_sum"],
                "decode_s": m["vllm:request_decode_time_seconds_sum"],
                "queue_s": m["vllm:request_queue_time_seconds_sum"],
                "e2e_s": m["vllm:e2e_request_latency_seconds_sum"],
                "itl_ms": itl_sum / itl_cnt * 1e3 if itl_cnt else None,
                "ttft_ms": (m["vllm:time_to_first_token_seconds_sum"] / m["vllm:time_to_first_token_seconds_count"]
                            * 1e3) if m["vllm:time_to_first_token_seconds_count"] else None,
                "prefix_cache_hit_rate": (m["vllm:prefix_cache_hits_total"] / m["vllm:prefix_cache_queries_total"])
                if m["vllm:prefix_cache_queries_total"] else None,
                "prompt_tokens": m["vllm:prompt_tokens_total"],
                "generation_tokens": m["vllm:generation_tokens_total"],
            },
            "per_question": qs,
        }
        v = run["vllm"]
        if v["itl_ms"]:
            v["eff_decode_gbps"] = args.weight_bytes / (v["itl_ms"] * 1e-3) / 1e9
            v["bw_util_of_spec"] = v["eff_decode_gbps"] / args.peak_gbps
        run["decode_share_of_agent_time"] = v["decode_s"] / sum(walls)
        run["prefill_share_of_agent_time"] = v["prefill_s"] / sum(walls)
        runs.append(run)

    summary = {"weight_bytes": args.weight_bytes, "peak_gbps": args.peak_gbps, "lpu_util": args.lpu_util,
               "by_concurrency": {}}
    for c in sorted({r["concurrency"] for r in runs}):
        rs = [r for r in runs if r["concurrency"] == c]

        def agg(f):
            xs = [f(r) for r in rs if f(r) is not None]
            return {"median": med(xs), "min": round(min(xs), 4), "max": round(max(xs), 4), "n": len(xs)} if xs else None

        s = {k: agg(lambda r, k=k: r[k]) for k in (
            "accuracy", "questions_per_min", "wall_p50_s", "wall_p95_s", "llm_share", "tool_share",
            "llm_calls_per_q", "prompt_tokens_per_q", "completion_tokens_per_q",
            "decode_share_of_agent_time", "prefill_share_of_agent_time")}
        for k in ("itl_ms", "ttft_ms", "prefix_cache_hit_rate", "eff_decode_gbps", "bw_util_of_spec", "queue_s"):
            s["vllm_" + k] = agg(lambda r, k=k: r["vllm"].get(k))
        # what-if: decode time scales with util_gpu / util_lpu at the same peak bandwidth (Amdahl on agent time)
        if s["vllm_bw_util_of_spec"] and s["decode_share_of_agent_time"]:
            u, f = s["vllm_bw_util_of_spec"]["median"], s["decode_share_of_agent_time"]["median"]
            s["what_if"] = {"gpu_util": u, "decode_share": f,
                            "agent_time_ratio": round(1 - f * (1 - u / args.lpu_util), 4)}
        summary["by_concurrency"][str(c)] = s

    json.dump({"summary": summary, "runs": runs}, open(os.path.join(args.results, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
