"""Write dataset.json: questions whose answers follow from the tools in graph.py."""
import json

from graph import CHIP_SPEC, MODEL_PARAMS_B

NAMES = {"llama-3.1-8b": "Llama-3.1-8B", "llama-3.1-70b": "Llama-3.1-70B", "qwen2.5-7b": "Qwen2.5-7B",
         "qwen2.5-1.5b": "Qwen2.5-1.5B", "mistral-7b": "Mistral-7B"}
CHIPS = {"rtx 3060 ti": "RTX 3060 Ti", "rtx 4090": "RTX 4090", "l4": "L4", "a100 80gb": "A100 80GB",
         "h100 sxm": "H100 SXM"}

rows = []


def add(q, value, unit):
    rows.append({"id": f"q{len(rows) + 1:02d}", "question": q, "answer": f"{value:.4g} {unit}",
                 "answer_value": value})


for m, c in [("llama-3.1-8b", "a100 80gb"), ("qwen2.5-7b", "l4"), ("mistral-7b", "rtx 4090"),
             ("llama-3.1-70b", "h100 sxm")]:
    p, (bw, _) = MODEL_PARAMS_B[m], CHIP_SPEC[c]
    add(f"What is the minimum decode time per token in milliseconds for {NAMES[m]} in fp16 on one "
        f"{CHIPS[c]}, if every token must read all weights once?", p * 2 / bw * 1000, "ms")

for m, c in [("qwen2.5-1.5b", "rtx 3060 ti"), ("llama-3.1-8b", "h100 sxm"), ("mistral-7b", "l4"),
             ("qwen2.5-7b", "a100 80gb")]:
    p, (bw, _) = MODEL_PARAMS_B[m], CHIP_SPEC[c]
    add(f"At batch size 1, what is the upper bound on decode tokens per second for {NAMES[m]} in fp16 "
        f"on a {CHIPS[c]}?", bw / (p * 2), "tokens/s")

for m, c in [("llama-3.1-8b", "rtx 4090"), ("qwen2.5-7b", "rtx 3060 ti"), ("llama-3.1-70b", "a100 80gb"),
             ("mistral-7b", "l4")]:
    p, (_, mem) = MODEL_PARAMS_B[m], CHIP_SPEC[c]
    add(f"How many GB of memory are left on a {CHIPS[c]} after loading {NAMES[m]} fp16 weights? "
        f"Give a negative number if it does not fit.", mem - p * 2, "GB")

for term, m, c in [("paged attention", "llama-3.1-8b", "l4"), ("kv cache", "qwen2.5-1.5b", "h100 sxm"),
                   ("continuous batching", "mistral-7b", "a100 80gb"),
                   ("speculative decoding", "qwen2.5-7b", "rtx 4090")]:
    p, (bw, _) = MODEL_PARAMS_B[m], CHIP_SPEC[c]
    add(f"First search what {term} is and summarize it in one sentence. Then compute the minimum decode "
        f"time per token in milliseconds for {NAMES[m]} in fp16 on a {CHIPS[c]}.", p * 2 / bw * 1000, "ms")

with open("dataset.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"{len(rows)} questions")
