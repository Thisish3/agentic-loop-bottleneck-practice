"""Does vLLM's reshape_and_cache_flash (the op that writes new K/V into the paged cache) produce the
same fp8 bytes as a plain torch cast? Layout as the FlashInfer backend allocates it: (blocks, 2, page, kv_heads, d)."""
import torch
import vllm._custom_ops as ops
torch.manual_seed(0)
dev = "cuda"
NB, PAGE, HKV, D, T = 8, 16, 4, 128, 100
for kv_dtype, f8 in (("fp8", torch.float8_e4m3fn), ("fp8_e5m2", torch.float8_e5m2)):
    k = torch.randn(T, HKV, D, dtype=torch.float16, device=dev)
    v = torch.randn(T, HKV, D, dtype=torch.float16, device=dev)
    cache = torch.zeros(NB, 2, PAGE, HKV, D, dtype=torch.uint8, device=dev)
    kc, vc = cache.unbind(1)
    slots = torch.randperm(NB * PAGE, device=dev)[:T].to(torch.int64)
    one = torch.tensor(1.0, dtype=torch.float32, device=dev)
    ops.reshape_and_cache_flash(k, v, kc, vc, slots, kv_dtype, one, one)
    got_k = kc.view(f8).reshape(NB * PAGE, HKV, D)[slots].float()
    got_v = vc.view(f8).reshape(NB * PAGE, HKV, D)[slots].float()
    exp_k, exp_v = k.to(f8).float(), v.to(f8).float()
    print(f"{kv_dtype}: K bytes equal {torch.equal(got_k, exp_k)}  max|diff| {(got_k-exp_k).abs().max().item():.4f}  "
          f"V equal {torch.equal(got_v, exp_v)}  rel err vs fp16 K {((got_k-k.float()).norm()/k.float().norm()).item():.4f}")
