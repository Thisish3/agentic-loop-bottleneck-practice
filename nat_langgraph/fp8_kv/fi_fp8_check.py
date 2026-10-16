"""Does FlashInfer's fp8-KV attention match fp16 attention over the *same* (dequantized) KV values?
Any difference here is kernel error, not quantization error. Qwen2.5-7B shapes: 28 q heads, 4 kv heads, d=128."""
import torch, flashinfer
torch.manual_seed(0)
dev = "cuda"
HQ, HKV, D, PAGE = 28, 4, 128, 16
lens = [700, 1300, 37, 2048]          # sequence lengths in the batch
pages = [(l + PAGE - 1) // PAGE for l in lens]
indptr = torch.tensor([0] + list(torch.tensor(pages).cumsum(0)), dtype=torch.int32, device=dev)
indices = torch.arange(sum(pages), dtype=torch.int32, device=dev)
last = torch.tensor([(l - 1) % PAGE + 1 for l in lens], dtype=torch.int32, device=dev)
kv16 = torch.randn(sum(pages), 2, PAGE, HKV, D, dtype=torch.float16, device=dev)
ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)

def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()

for f8 in (torch.float8_e4m3fn, torch.float8_e5m2):
    kv8 = kv16.to(f8)
    ref_kv = kv8.to(torch.float16)        # exactly the values the fp8 cache holds
    # decode: one query token per sequence
    q = torch.randn(len(lens), HQ, D, dtype=torch.float16, device=dev)
    outs = {}
    for name, cache in (("fp16", ref_kv), ("fp8", kv8)):
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)  # GQA group 7: vLLM uses the tensor-core path too
        w.plan(indptr, indices, last, HQ, HKV, D, PAGE, q_data_type=torch.float16, kv_data_type=cache.dtype)
        outs[name] = w.run(q, cache)
    print(f"{f8}: decode  rel err fp8-kernel vs fp16-kernel = {rel(outs['fp8'], outs['fp16']):.4f}")
    # prefill: 32 new query tokens per sequence, causal against the paged cache
    qlen = 32
    qo = torch.tensor([0] + [qlen * (i + 1) for i in range(len(lens))], dtype=torch.int32, device=dev)
    qp = torch.randn(qlen * len(lens), HQ, D, dtype=torch.float16, device=dev)
    for name, cache in (("fp16", ref_kv), ("fp8", kv8)):
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        w.plan(qo, indptr, indices, last, HQ, HKV, D, PAGE, causal=True,
               q_data_type=torch.float16, kv_data_type=cache.dtype)
        outs[name] = w.run(qp, cache)
    print(f"{f8}: prefill rel err fp8-kernel vs fp16-kernel = {rel(outs['fp8'], outs['fp16']):.4f}")
print("reference: rel err of fp16 kernel vs itself rerun = 0; a correct fp8 kernel should be ~1e-3")
