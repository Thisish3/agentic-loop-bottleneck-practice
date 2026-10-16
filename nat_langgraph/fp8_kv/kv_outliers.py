"""Capture real Q/K/V of Qwen2.5-7B-AWQ (fp16 KV, eager) for one chat prompt, then measure per layer how much
fp8 quantization of K/V alone (scale 1.0, as vLLM uses without checkpoint scales) perturbs the attention output."""
import json, os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams
from vllm.attention.layer import Attention

llm = LLM("Qwen/Qwen2.5-7B-Instruct-AWQ", max_model_len=1024, gpu_memory_utilization=0.85, max_num_seqs=4, swap_space=0,
          enforce_eager=True, enable_prefix_caching=False)
model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model
caps = {}
def hook(name):
    def f(mod, args, kwargs):
        q, k, v = args[:3]
        if q.shape[0] > 8 and name not in caps:   # the prefill pass of our prompt
            caps[name] = (q.detach().float().cpu(), k.detach().float().cpu(), v.detach().float().cpu())
    return f
for n, m in model.named_modules():
    if isinstance(m, Attention):
        m.register_forward_pre_hook(hook(n), with_kwargs=True)
msg = [{"role": "user", "content": "What is 17*23? Explain in two sentences."}]
llm.chat(msg, SamplingParams(max_tokens=1, temperature=0))

HQ, HKV, D = 28, 4, 128
def attn(q, k, v):  # causal GQA attention, fp32
    T = q.shape[0]
    q = q.view(T, HQ, D).transpose(0, 1)
    k = k.view(T, HKV, D).transpose(0, 1).repeat_interleave(HQ // HKV, 0)
    v = v.view(T, HKV, D).transpose(0, 1).repeat_interleave(HQ // HKV, 0)
    s = q @ k.transpose(1, 2) / D ** 0.5
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    return (s.softmax(-1) @ v)
def rt(x, f8):  # what an unscaled fp8 cache hands back
    return x.half().to(f8).float()

rows = []
for i, (name, (q, k, v)) in enumerate(caps.items()):
    ref = attn(q, k, v)
    r = {"layer": i, "max_abs_k": k.abs().max().item(), "max_abs_v": v.abs().max().item(),
         "k_over_448": (k.abs() > 448).float().mean().item()}
    for tag, f8 in (("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)):
        out = attn(q, rt(k, f8), rt(v, f8))
        r[f"err_{tag}"] = ((out - ref).norm() / ref.norm()).item()
        r[f"nan_{tag}"] = bool(torch.isnan(out).any())
    # per-tensor scale (what --calculate-kv-scales / a calibrated checkpoint would give)
    ks, vs = k.abs().max() / 448, v.abs().max() / 448
    out = attn(q, rt(k / ks, torch.float8_e4m3fn) * ks, rt(v / vs, torch.float8_e4m3fn) * vs)
    r["err_e4m3_scaled"] = ((out - ref).norm() / ref.norm()).item()
    # per-channel K scale (one scale per head-dim channel, as KIVI-style K quantization does)
    kc = k.view(-1, HKV, D).abs().amax(0, keepdim=True) / 448
    kq = (rt(k.view(-1, HKV, D) / kc, torch.float8_e4m3fn) * kc).view_as(k)
    out = attn(q, kq, rt(v / vs, torch.float8_e4m3fn) * vs)
    r["err_e4m3_chan"] = ((out - ref).norm() / ref.norm()).item()
    # Oaken-style split: the largest-magnitude 1% / 5% of K and V entries stay fp16, the rest go to e4m3
    for frac in (0.01, 0.05):
        def split(x):
            thr = x.abs().flatten().kthvalue(int(x.numel() * (1 - frac))).values
            big = x.abs() > thr
            return torch.where(big, x, rt(x, torch.float8_e4m3fn))
        out = attn(q, split(k), split(v))
        r[f"err_e4m3_keep{int(frac*100)}"] = ((out - ref).norm() / ref.norm()).item()
    rows.append(r)
print(f"tokens in prompt: {next(iter(caps.values()))[0].shape[0]}, layers captured: {len(rows)}")
print("layer  max|K|   max|V|   e4m3   e5m2  e4m3_tensor  e4m3_chanK  keep1%  keep5%")
for r in rows:
    print(f"{r['layer']:5d} {r['max_abs_k']:7.1f} {r['max_abs_v']:7.1f}  {r['err_e4m3']:.3f}  {r['err_e5m2']:.3f}"
          f"     {r['err_e4m3_scaled']:.3f}       {r['err_e4m3_chan']:.3f}    {r['err_e4m3_keep1']:.3f}   {r['err_e4m3_keep5']:.3f}")
import statistics as st
for key in ("err_e4m3", "err_e5m2", "err_e4m3_scaled", "err_e4m3_chan", "err_e4m3_keep1", "err_e4m3_keep5"):
    print(f"median {key}: {st.median(r[key] for r in rows):.4f}  max: {max(r[key] for r in rows):.4f}")
json.dump(rows, open(sys.argv[1], "w"), indent=1)
