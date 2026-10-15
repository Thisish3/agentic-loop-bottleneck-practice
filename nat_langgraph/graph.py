"""LangGraph tool-calling agent served by vLLM, wrapped by NeMo Agent Toolkit.

NAT loads this file through `langgraph_wrapper` (config.yml: graph: .../graph.py:graph).
The LLM is a vLLM OpenAI-compatible server (VLLM_BASE_URL, VLLM_MODEL).

Tools are deterministic so every question has a computable answer:
  get_model_params   parameter count of an LLM
  get_chip_spec      memory bandwidth / capacity of an accelerator
  calculator         arithmetic
  web_search         canned snippets behind a fixed 0.5 s delay (stands in for a slow external API)
"""

import ast
import operator
import os
import time

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

MODEL_PARAMS_B = {
    "llama-3.1-8b": 8.03,
    "llama-3.1-70b": 70.6,
    "qwen2.5-7b": 7.62,
    "qwen2.5-1.5b": 1.54,
    "mistral-7b": 7.25,
}

CHIP_SPEC = {  # memory bandwidth (GB/s), memory (GB)
    "rtx 3060 ti": (448, 8),
    "rtx 4090": (1008, 24),
    "l4": (300, 24),
    "a100 80gb": (2039, 80),
    "h100 sxm": (3350, 80),
}

SEARCH = {
    "kv cache": "The KV cache stores past keys and values so decode only computes attention for the new token.",
    "continuous batching": "Continuous batching adds and removes requests from the running batch every step.",
    "paged attention": "PagedAttention stores the KV cache in fixed-size blocks, like virtual memory pages.",
    "speculative decoding": "Speculative decoding drafts several tokens cheaply and verifies them in one pass.",
}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only + - * / ** and numbers are allowed")


def _key(name: str, table: dict) -> str | None:
    n = name.lower().replace("_", " ").replace("meta-", "").strip()
    return next((k for k in table if k in n or n in k), None)


@tool
def get_model_params(model: str) -> str:
    """Return the parameter count (in billions) of an LLM, e.g. 'Llama-3.1-8B'."""
    k = _key(model.replace(" ", "-"), MODEL_PARAMS_B)
    return f"{k}: {MODEL_PARAMS_B[k]} billion parameters" if k else f"unknown model: {model}"


@tool
def get_chip_spec(chip: str) -> str:
    """Return memory bandwidth (GB/s) and memory size (GB) of a GPU/accelerator, e.g. 'A100 80GB'."""
    k = _key(chip, CHIP_SPEC)
    if not k:
        return f"unknown chip: {chip}"
    bw, mem = CHIP_SPEC[k]
    return f"{k}: memory bandwidth {bw} GB/s, memory {mem} GB"


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression with + - * / ** and parentheses, e.g. '8.03e9*2/2039e9*1000'."""
    try:
        return str(round(_eval(ast.parse(expression, mode="eval").body), 6))
    except Exception as e:  # returned to the model so it can retry
        return f"error: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web for a short definition of an LLM-serving term."""
    time.sleep(0.5)
    k = _key(query, SEARCH)
    return SEARCH[k] if k else "no result"


SYSTEM = (
    "You are a precise assistant for LLM inference questions. Always use the tools to look up "
    "model sizes and chip specs, and use the calculator for every arithmetic step; never guess numbers. "
    "Assume fp16 weights take 2 bytes per parameter and that every decode step reads all weights once. "
    "End with a short final answer that contains the number and its unit."
)


def graph(_config=None):
    llm = ChatOpenAI(
        base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key="not-needed",
        model=os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        temperature=0,
        max_tokens=512,
    )
    return create_react_agent(llm, [get_model_params, get_chip_spec, calculator, web_search], prompt=SYSTEM)
