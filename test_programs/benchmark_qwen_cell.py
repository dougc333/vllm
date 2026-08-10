"""
Single-cell Colab benchmark: Qwen3-0.6B raw TTFT p95/p99.
No vLLM, no nanovLLM — pure transformers pipeline. ~30 loc.

Copy-paste this ENTIRE block into one Colab cell and run.
Works locally too: python benchmark_qwen_cell.py
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "transformers"])

import time, numpy as np, torch
from transformers import pipeline

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "Explain Python lists vs tuples:",
    "Write a palindrome checker:",
    "What is the capital of France?",
    "Convert 150C to Fahrenheit:",
    "Three ways to reduce carbon emissions:",
    "Haiku about programming:",
    "Explain quantum computing simply:",
    "REST vs GraphQL differences:",
    "How does garbage collection work?",
    "Summarize Romeo and Juliet:",
] * 5  # 50 total

pipe = pipeline("text-generation", model=MODEL, device=0, torch_dtype=torch.float16)

ttfts, itls = [], []
for i, p in enumerate(PROMPTS):
    start = time.perf_counter()
    out = pipe(p, max_new_tokens=1, do_sample=False, return_full_text=False)
    ttft = time.perf_counter() - start
    ttfts.append(ttft)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(PROMPTS)} - TTFT: {ttft*1000:.1f} ms")

# Decode ITL: one request with longer output
start = time.perf_counter()
out = pipe(PROMPTS[0], max_new_tokens=64, do_sample=False, return_full_text=False)
wall = time.perf_counter() - start
n_tok = len(out[0]["generated_token_ids"])
itl = (wall / max(n_tok, 1)) * 1000  # ms/token

def pctl(v, q):
    xs = sorted(v)
    k = (len(xs)-1) * q/100
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f) if c != f else xs[f]

print("\n" + "="*50)
print(f"  Qwen3-0.6B - HF pipeline (FP16)")
print("="*50)
print(f"  TTFT p50: {pctl(ttfts, 50)*1000:.1f} ms")
print(f"  TTFT p90: {pctl(ttfts, 90)*1000:.1f} ms")
print(f"  TTFT p95: {pctl(ttfts, 95)*1000:.1f} ms")
print(f"  TTFT p99: {pctl(ttfts, 99)*1000:.1f} ms")
print(f"  TTFT mean: {np.mean(ttfts)*1000:.1f} ms")
print(f"  Decode ITL: {itl:.2f} ms/token")
print(f"  Throughput: {1/itl*1000:.1f} tok/s")
print("="*50)