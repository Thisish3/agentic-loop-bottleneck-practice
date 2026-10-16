"""vLLM general plugin: emulate an fp8 (e4m3, scale 1.0) KV cache per layer on top of an fp16 cache.

vLLM 0.11 has one kv_cache_dtype for every layer, so mixed precision cannot be configured directly. This plugin wraps
Attention.forward and rounds K/V through float8_e4m3fn right before they are written to the cache and attended to.
The rounding is the same cast vLLM's fp8 cache write performs (byte-identical, see cache_write_check.py), so the
*accuracy* of a mixed fp8/fp16 cache is reproduced exactly; the memory saving is not (the cache stays fp16).

Environment (read in every vLLM process):
  KV_EMU=off|e4m3               off: wrapper installed but does nothing (control)
  KV_EMU_KEEP_LAYERS=0,27       these layers keep fp16 K/V
  KV_EMU_TOP=0.05               per token, the largest-magnitude 5% of K (and of V) entries keep fp16
"""
import os
import re

_installed = False


def _parse():
    mode = os.environ.get("KV_EMU", "off")
    keep = {int(x) for x in os.environ.get("KV_EMU_KEEP_LAYERS", "").split(",") if x.strip()}
    top = float(os.environ.get("KV_EMU_TOP", "0") or 0)
    return mode, keep, top


def register():
    global _installed
    if _installed:
        return
    _installed = True
    import torch
    from vllm.attention.layer import Attention

    mode, keep, top = _parse()
    f8 = torch.float8_e4m3fn
    orig = Attention.forward
    seen = set()

    def emu(x):
        y = x.to(f8).to(x.dtype)
        if top > 0:
            k = max(1, int(round(top * x.shape[-1])))
            thr = x.abs().topk(k, dim=-1).values[..., -1:]
            y = torch.where(x.abs() >= thr, x, y)
        return y

    def forward(self, query, key, value, *args, **kwargs):
        if mode == "e4m3":
            idx = getattr(self, "_kv_emu_idx", None)
            if idx is None:
                m = re.search(r"layers\.(\d+)\.", self.layer_name)
                idx = self._kv_emu_idx = int(m.group(1)) if m else -1
            if idx not in keep:
                key, value = emu(key), emu(value)
            if idx not in seen:
                seen.add(idx)
                if len(seen) == 1 or idx in keep:
                    print(f"[kv_emu] layer {idx}: {'fp16 (kept)' if idx in keep else 'e4m3'} top={top}", flush=True)
        return orig(self, query, key, value, *args, **kwargs)

    Attention.forward = forward
    print(f"[kv_emu] installed pid={os.getpid()} mode={mode} keep={sorted(keep)} top={top}", flush=True)
