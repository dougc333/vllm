"""
nanovLLM — minimal LLM serving engine with PagedAttention KV cache + continuous batching.
~500 lines, educational, no vLLM dependency. Runs on a single Colab T4 GPU.

Usage (Colab cell):
  !pip install transformers accelerate
  %run benchmark_qwen.py

Optional: swap MODEL to any HuggingFace model.
"""
from __future__ import annotations

import time, math, asyncio, argparse, json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══════════════════════════════════════════════════════════
# Part 1 — KV Cache Block Manager (PagedAttention-style)
# ═══════════════════════════════════════════════════════════

@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 16          # tokens per block
    num_gpu_blocks: int = 2048    # total blocks on GPU

class BlockManager:
    """Manages a pool of KV cache blocks. Each block stores `block_size`
    tokens worth of key/value tensors for every layer and head."""

    def __init__(self, config: KVCacheConfig, dtype: torch.dtype, device: torch.device):
        self.cfg = config
        self.dtype = dtype
        self.device = device
        # Shape: [num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim]
        #   dim 1: 0=key, 1=value
        self.pool = torch.zeros(
            config.num_gpu_blocks, 2, config.num_layers,
            config.num_kv_heads, config.block_size, config.head_dim,
            dtype=dtype, device=device,
        )
        self.free_blocks = deque(range(config.num_gpu_blocks))

    def alloc(self, n: int = 1) -> list[int]:
        return [self.free_blocks.popleft() for _ in range(n)]

    def free(self, blocks: list[int]):
        for b in blocks:
            self.free_blocks.append(b)

    def get_k(self, block_id: int, layer: int) -> torch.Tensor:
        return self.pool[block_id, 0, layer]  # [num_kv_heads, block_size, head_dim]

    def get_v(self, block_id: int, layer: int) -> torch.Tensor:
        return self.pool[block_id, 1, layer]  # [num_kv_heads, block_size, head_dim]

# ═══════════════════════════════════════════════════════════
# Part 2 — Request and Sequence
# ═══════════════════════════════════════════════════════════

@dataclass
class Request:
    req_id: str
    prompt: str
    prompt_ids: list[int] = field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.0

@dataclass
class Sequence:
    req: Request
    block_ids: list[int] = field(default_factory=list)
    num_cached: int = 0            # tokens already in KV cache
    num_output: int = 0            # tokens generated so far
    done: bool = False
    ttft_s: Optional[float] = None # time-to-first-token
    _start_ts: float = 0.0         # wall-clock when first scheduled

# ═══════════════════════════════════════════════════════════
# Part 3 — Model Runner (single forward pass)
# ═══════════════════════════════════════════════════════════

class ModelRunner:
    """Wraps a HF model. Runs one batch of sequences through the model,
    writing new KV cache into allocated blocks."""

    def __init__(self, model: nn.Module, tokenizer, block_mgr: BlockManager,
                 device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.block_mgr = block_mgr
        self.device = device

    @torch.no_grad()
    def run_batch(self, seqs: list[Sequence]) -> dict[str, int]:
        """Run one decoding step. Returns {req_id: next_token_id}."""
        if not seqs:
            return {}

        cfg = self.block_mgr.cfg
        batch_size = len(seqs)

        # Determine which tokens need processing
        max_prompt = max(len(s.req.prompt_ids) for s in seqs)
        max_kv = max(s.num_cached for s in seqs) if any(s.num_cached > 0 for s in seqs) else 0
        total_len = max(max_prompt, max_kv + 1)  # +1 for the new token position

        # Build input_ids [batch, total_len]
        input_ids = torch.full((batch_size, total_len), self.tokenizer.pad_token_id or 0,
                               dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            if s.num_cached == 0:
                # Prefill — need all prompt tokens
                ids = s.req.prompt_ids[:total_len]
                input_ids[i, :len(ids)] = torch.tensor(ids, device=self.device)
            else:
                # Decode — only the last generated token
                last_id = s.req.prompt_ids[s.num_cached] if s.num_cached < len(s.req.prompt_ids) else s.req.prompt_ids[-1]
                input_ids[i, s.num_cached] = last_id

        # Attention mask: mask out padding
        attn_mask = torch.ones(batch_size, total_len, device=self.device)
        for i, s in enumerate(seqs):
            if s.num_cached == 0:
                attn_mask[i, len(s.req.prompt_ids):] = 0

        # Build past_key_values from KV cache blocks
        past_kv: Optional[list] = None
        if max_kv > 0:
            past_kv = []
            for layer_idx in range(cfg.num_layers):
                keys_list, values_list = [], []
                for s in seqs:
                    if s.num_cached > 0:
                        full_blocks = s.num_cached // cfg.block_size
                        partial = s.num_cached % cfg.block_size
                        k_parts = []
                        v_parts = []
                        for j in range(full_blocks):
                            k_parts.append(self.block_mgr.get_k(s.block_ids[j], layer_idx)[:cfg.block_size])
                            v_parts.append(self.block_mgr.get_v(s.block_ids[j], layer_idx)[:cfg.block_size])
                        if partial > 0 and full_blocks < len(s.block_ids):
                            k_parts.append(self.block_mgr.get_k(s.block_ids[full_blocks], layer_idx)[:partial])
                            v_parts.append(self.block_mgr.get_v(s.block_ids[full_blocks], layer_idx)[:partial])
                        if k_parts:
                            keys_list.append(torch.cat(k_parts, dim=0))
                            values_list.append(torch.cat(v_parts, dim=0))
                        else:
                            keys_list.append(torch.zeros(0, cfg.head_dim, device=self.device, dtype=self.block_mgr.dtype))
                            values_list.append(torch.zeros(0, cfg.head_dim, device=self.device, dtype=self.block_mgr.dtype))
                    else:
                        keys_list.append(torch.zeros(0, cfg.head_dim, device=self.device, dtype=self.block_mgr.dtype))
                        values_list.append(torch.zeros(0, cfg.head_dim, device=self.device, dtype=self.block_mgr.dtype))

                max_len = max(k.shape[0] for k in keys_list)
                pad_k = torch.zeros(batch_size, cfg.num_kv_heads, max_len, cfg.head_dim,
                                    device=self.device, dtype=self.block_mgr.dtype)
                pad_v = torch.zeros_like(pad_k)
                for i, (k, v) in enumerate(zip(keys_list, values_list)):
                    if k.shape[0] > 0:
                        pad_k[i, :, :k.shape[0], :] = k.unsqueeze(0)
                        pad_v[i, :, :v.shape[0], :] = v.unsqueeze(0)
                past_kv.append((pad_k, pad_v))
            past_kv = tuple(past_kv)

        # Forward pass
        if past_kv is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]  # [batch, vocab]
        next_tokens = logits.argmax(dim=-1).cpu().tolist()

        # Write new KV cache slots
        past_kv_raw = outputs.past_key_values
        for i, s in enumerate(seqs):
            slot_in_block = s.num_cached % cfg.block_size
            if slot_in_block == 0:
                new_blocks = self.block_mgr.alloc(1)
                s.block_ids.extend(new_blocks)
            block_id = s.block_ids[-1]
            for layer_idx, (k, v) in enumerate(past_kv_raw):
                self.block_mgr.pool[block_id, 0, layer_idx, :, slot_in_block, :] = k[i, :, -1, :]
                self.block_mgr.pool[block_id, 1, layer_idx, :, slot_in_block, :] = v[i, :, -1, :]
            s.num_cached += 1

        return {s.req.req_id: tok for s, tok in zip(seqs, next_tokens)}

# ═══════════════════════════════════════════════════════════
# Part 4 — Scheduler (FIFO, max batch)
# ═══════════════════════════════════════════════════════════

class Scheduler:
    """Simple FIFO scheduler with max batch size."""

    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def step(self) -> list[Sequence]:
        while self.waiting and len(self.running) < self.max_batch_size:
            seq = self.waiting.popleft()
            seq._start_ts = time.perf_counter()
            self.running.append(seq)
        active = [s for s in self.running if not s.done]
        return active[:self.max_batch_size]

    def mark_done(self, seq: Sequence):
        seq.done = True
        self.finished.append(seq)
        if seq in self.running:
            self.running.remove(seq)

# ═══════════════════════════════════════════════════════════
# Part 5 — NanoEngine (orchestrates the loop)
# ═══════════════════════════════════════════════════════════

class NanoEngine:
    """Minimal async serving engine."""

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B",
                 max_batch_size: int = 8, block_size: int = 16,
                 gpu_blocks: int = 2048, max_tokens: int = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.max_tokens = max_tokens

        print(f"Loading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self.dtype, trust_remote_code=True,
            device_map="auto",
        ).eval()

        # Detect KV cache config from model
        cfg = self.model.config
        num_layers = cfg.num_hidden_layers
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = cfg.hidden_size // cfg.num_attention_heads

        kv_config = KVCacheConfig(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_gpu_blocks=gpu_blocks,
        )
        self.block_mgr = BlockManager(kv_config, self.dtype, self.device)
        self.runner = ModelRunner(self.model, self.tokenizer, self.block_mgr, self.device)
        self.scheduler = Scheduler(max_batch_size=max_batch_size)
        self._next_id = 0

    def add_request(self, prompt: str, max_tokens: int = 512) -> str:
        rid = f"req-{self._next_id}"
        self._next_id += 1
        req = Request(req_id=rid, prompt=prompt, max_tokens=max_tokens)
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        req.prompt_ids = ids[:4096]
        self.scheduler.add(Sequence(req=req))
        return rid

    @torch.no_grad()
    def step(self):
        batch = self.scheduler.step()
        if not batch:
            return
        outputs = self.runner.run_batch(batch)
        for s in batch:
            rid = s.req.req_id
            tok = outputs.get(rid)
            if tok is not None:
                s.num_output += 1
                if s.num_output == 1 and s.ttft_s is None:
                    s.ttft_s = time.perf_counter() - s._start_ts
                s.req.prompt_ids.append(tok)
                if tok == self.tokenizer.eos_token_id:
                    self.scheduler.mark_done(s)
            if s.num_output >= s.req.max_tokens:
                self.scheduler.mark_done(s)

    async def step_async(self):
        self.step()
        await asyncio.sleep(0)

    def is_busy(self) -> bool:
        return bool(self.scheduler.running) or bool(self.scheduler.waiting)

# ═══════════════════════════════════════════════════════════
# Part 6 — Benchmark
# ═══════════════════════════════════════════════════════════

def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if c == f:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

PROMPTS = [
    "Explain the difference between a list and a tuple in Python:",
    "Write a function that checks if a string is a palindrome:",
    "What is the capital of France?",
    "Convert 150 degrees Celsius to Fahrenheit and show your work:",
    "Describe three ways to reduce carbon emissions:",
    "Write a haiku about programming:",
    "Explain quantum computing in simple terms:",
    "What are the key differences between REST and GraphQL?",
    "How does garbage collection work in modern programming languages?",
    "Summarize the plot of Romeo and Juliet in one paragraph:",
] * 5  # 50 requests

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-requests", type=int, default=50)
    args = parser.parse_args()

    prompts = PROMPTS[:args.num_requests]

    engine = NanoEngine(
        model_id=args.model,
        max_batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )

    t_start = time.perf_counter()
    for prompt in prompts:
        engine.add_request(prompt, max_tokens=args.max_tokens)

    step_count = 0
    while engine.is_busy():
        await engine.step_async()
        step_count += 1

    t_end = time.perf_counter()
    total_wall = t_end - t_start

    ttfbs = []
    total_output_tokens = 0
    for seq in engine.scheduler.finished:
        if seq.ttft_s is not None:
            ttfbs.append(seq.ttft_s)
        total_output_tokens += seq.num_output

    p = [50, 90, 95, 99]
    print("\n" + "=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Engine: nanovLLM (PagedAttention + continuous batching)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Requests: {len(prompts)}  |  Steps: {step_count}")
    print(f"  Wall time: {total_wall:.1f}s")
    print("=" * 60)
    print(f"  {'Metric':<22} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'TTFT (ms)':<22} " + " ".join(f"{percentile(ttfbs, q)*1000:>8.1f}" for q in p))
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  Total output tokens:   {total_output_tokens}")
    print(f"  Throughput:            {total_output_tokens/total_wall:.1f} tok/s")
    print(f"  Mean TTFT:             {sum(ttfbs)/len(ttfbs)*1000:.1f} ms")
    print(f"  Mean Decode ITL:       {total_wall / max(total_output_tokens, 1) * 1000:.2f} ms/tok")
    print("=" * 60)

    result = {
        "model": args.model,
        "engine": "nanovLLM",
        "num_requests": len(prompts),
        "batch_size": args.batch_size,
        "wall_s": total_wall,
        "total_output_tokens": total_output_tokens,
        "throughput_tok_per_s": total_output_tokens / total_wall,
        "ttft_ms": {f"p{q}": percentile(ttfbs, q) * 1000 for q in p},
        "ttft_mean_ms": sum(ttfbs) / len(ttfbs) * 1000 if ttfbs else 0,
    }
    print("\n" + json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())