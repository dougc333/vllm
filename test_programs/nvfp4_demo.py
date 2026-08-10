"""
Dynamic NVFP4 Quantization — Educational Implementation.
Implements Unsloth-style 4-bit FP quantization from scratch on a tiny SmolLM model.

What you'll learn:
  1. What NVFP4 is (FP4 E2M1 format + dynamic per-group scaling)
  2. How quantize/dequantize works
  3. Memory savings (4-bit vs 16-bit)
  4. Accuracy loss measurement
  5. The fused dequantize-matmul concept (the "secret sauce")

Run:  python nvfp4_demo.py
Colab: just paste the whole file into a cell
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch", "transformers", "accelerate"])

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# ═══════════════════════════════════════════════════════════════
# PART 1 — FP4 Format: E2M1 (1 sign, 2 exponent, 1 mantissa)
# ═══════════════════════════════════════════════════════════════

def fp4_e2m1_table() -> list[float]:
    """The 16 values representable in E2M1 FP4 format.

    E2M1: 1 sign + 2 exponent (bias 1) + 1 mantissa
    Exponent: 0=subnormal, 1, 2, 3=inf/NaN
    Mantissa: {0, 0.5} (implicit 1 bit for normal, 0 for subnormal)
    """
    table = []
    for sign in [1.0, -1.0]:
        for exp in range(4):           # 2 bits → 0..3
            bias = 1
            actual_exp = exp - bias     # -1, 0, 1, 2
            if exp == 0:
                # Subnormal: implicit bit = 0, mantissa = 0 or 0.5
                for m in [0.0, 0.5]:
                    value = m * (2 ** (actual_exp + 1))
                    table.append(sign * value)
            elif exp == 3:
                # inf/NaN — skip (treat as max norm)
                pass
            else:
                # Normal: implicit bit = 1
                for m in [0.0, 0.5]:
                    value = (1.0 + m) * (2 ** actual_exp)
                    table.append(sign * value)
    # Exclude -0 (it's the same as 0 in practice)
    values = sorted(set(table))  # deduplicate +0 / -0
    return values

FP4_VALUES = torch.tensor([  # E2M1 representable grid, sorted
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5,
    0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
], dtype=torch.float16)

# Simpler: E2M1 has 8 positive + 8 negative + 0 = 16-17 distinct values
# We'll use a 16-entry lookup for packing: index 0..15
FP4_QUANTIZED = FP4_VALUES

@torch.inference_mode()
def quantize_group(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a group of values to FP4 using nearest-neighbor with dynamic scale.

    Args:
        values: [group_size] FP16 tensor
    Returns:
        packed: [ceil(group_size/2)] uint8 tensor (2 values per byte)
        scale: FP32 scalar — the max absolute value in the group
    """
    scale = values.abs().max().float()
    if scale < 1e-12:
        # All zeros — pack as zeros with scale = 0
        packed = torch.zeros((values.numel() + 1) // 2, dtype=torch.uint8, device=values.device)
        return packed, scale

    # Scale values to the FP4 representable range [-6, 6]
    # scale_factor maps group max → max representable (6.0)
    scaled = values.float() / scale  # normalize to [-1, 1]
    scaled = scaled * 6.0            # scale to FP4 range

    # Find nearest FP4 representable value for each scaled value
    # FP4_VALUES is on CPU; we use a simple loop approach for clarity
    fp4_cpu = FP4_QUANTIZED.float().cpu()
    val_cpu = scaled.cpu()
    indices_list = [(val_cpu - fv).abs().argmin().item() for fv in fp4_cpu.unsqueeze(1)]
    # Actually compute properly:
    indices = []
    for v in val_cpu:
        idx = (v.unsqueeze(-1) - fp4_cpu).abs().argmin().item()
        indices.append(idx % 16)  # ensure in [0, 15]

    # Pack: every 2 indices → 1 byte (nibble1 | nibble2 << 4)
    n = len(indices)
    packed_list = []
    for i in range(0, n, 2):
        nib0 = indices[i]
        nib1 = indices[i + 1] if i + 1 < n else 0
        packed_list.append(nib0 | (nib1 << 4))
    packed = torch.tensor(packed_list, dtype=torch.uint8)
    return packed, scale


@torch.inference_mode()
def dequantize_group(packed: torch.Tensor, scale: torch.Tensor, num_vals: int) -> torch.Tensor:
    """Dequantize a group from FP4 back to FP16.

    Args:
        packed: [ceil(num_vals/2)] uint8 (each byte = 2 nibble-index values)
        scale: FP32 scalar
        num_vals: original number of values
    Returns:
        values: [num_vals] FP16 tensor
    """
    indices_list = []
    for byte_val in packed.cpu().tolist():
        indices_list.append(byte_val & 0x0F)       # lower nibble
        indices_list.append((byte_val >> 4) & 0x0F)  # upper nibble
    indices = torch.tensor(indices_list[:num_vals], dtype=torch.long)

    fp4_cpu = FP4_QUANTIZED.float()
    scaled = fp4_cpu[indices]  # values in FP4 range [-6, 6]
    # Rescale back: the inverse of quantize_group scaling
    values = (scaled / 6.0) * scale.float()
    return values.half()


@torch.inference_mode()
def quantize_fp4(weights: torch.Tensor, group_size: int = 32) -> tuple[list[torch.Tensor], list[torch.Tensor], tuple[int, ...]]:
    """Quantize a weight matrix to FP4 with per-group dynamic scaling.

    Args:
        weights: [rows, cols] FP16 tensor
        group_size: number of elements per scale group (default 32)
    Returns:
        packed_groups: list of group-packed nibble tensors
        scales: list of FP32 scale tensors
        original_shape: shape for reconstruction
    """
    flat = weights.flatten()
    packed_groups, scales = [], []
    for start in range(0, flat.numel(), group_size):
        group = flat[start:start + group_size]
        pk, sc = quantize_group(group)
        packed_groups.append(pk)
        scales.append(sc)
    return packed_groups, scales, weights.shape


@torch.inference_mode()
def dequantize_fp4(packed_groups: list[torch.Tensor], scales: list[torch.Tensor],
                   original_shape: tuple[int, ...], group_size: int = 32) -> torch.Tensor:
    """Reconstruct weights from FP4 quantization."""
    total_vals = original_shape[0] * original_shape[1] if len(original_shape) >= 2 else original_shape[0]
    vals_list = []
    for start in range(0, total_vals, group_size):
        idx = start // group_size
        num_vals = min(group_size, total_vals - start)
        vals = dequantize_group(packed_groups[idx], scales[idx], num_vals)
        vals_list.append(vals)
    flat = torch.cat(vals_list)
    return flat.view(original_shape)


# ═══════════════════════════════════════════════════════════════
# PART 2 — Fused Dequantize-MatMul (the "secret sauce")
# ═══════════════════════════════════════════════════════════════

@torch.inference_mode()
def fused_dequant_matmul(x: torch.Tensor, packed_groups: list[torch.Tensor],
                         scales: list[torch.Tensor], shape: tuple[int, ...],
                         group_size: int = 32) -> torch.Tensor:
    """In a real kernel this would fuse dequantize + matmul into one kernel.
    Here we show what it *computes* — dequant on-the-fly, no full FP16 copy.
    """
    # In Triton/CUDA:
    #   for each weight row chunk in registers:
    #       read FP4 nibbles → dequant to FP16 in registers
    #       dot product with x in registers
    #       never write the full FP16 weight matrix to memory
    #
    # This is a naive Python equivalent for demonstration:
    w_fp16 = dequantize_fp4(packed_groups, scales, shape, group_size)
    return x @ w_fp16.to(x.dtype)


# ═══════════════════════════════════════════════════════════════
# PART 3 — Tiny SmolLM Model
# ═══════════════════════════════════════════════════════════════

class TinyAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)

class TinyFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class TinyBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = TinyAttention(d_model, n_heads)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = TinyFFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class TinyLM(nn.Module):
    """A tiny GPT-like LM for demonstration (~300K params)."""
    def __init__(self, vocab_size=1000, d_model=64, n_layers=3, n_heads=4, d_ff=256):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TinyBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, x):
        h = self.tok_embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h)

    def generate(self, prompt_ids, max_new=20, temperature=0.8):
        self.eval()
        ids = list(prompt_ids)
        with torch.no_grad():
            for _ in range(max_new):
                x = torch.tensor([ids[-64:]], dtype=torch.long)
                logits = self.forward(x)
                probs = F.softmax(logits[0, -1] / temperature, dim=0)
                next_id = torch.multinomial(probs, 1).item()
                ids.append(next_id)
                if next_id == 0:
                    break
        return ids


# ═══════════════════════════════════════════════════════════════
# PART 4 — Demo
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Dynamic NVFP4 Quantization — Educational Demo")
    print("=" * 65)

    # ── Create and train a tiny model ──
    torch.manual_seed(42)
    model = TinyLM()
    trainer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Quick training on synthetic data (just to get non-random weights)
    print("\n  Training tiny LM (100 steps)...")
    for step in range(100):
        x = torch.randint(0, 1000, (4, 32))
        loss = F.cross_entropy(model(x).reshape(-1, 1000), x.reshape(-1))
        trainer.zero_grad(); loss.backward(); trainer.step()
        if (step + 1) % 25 == 0:
            print(f"    step {step+1:3d}  loss = {loss.item():.3f}")

    # ── Measure FP16 baseline ──
    prompt = [5, 20, 100, 300, 50]
    pred_fp16 = model.generate(prompt, max_new=30)
    print(f"\n  FP16 generate (seed 5,20,100,300,50): {pred_fp16[:10]}...")

    # ── Quantize all linear layers to NVFP4 ──
    print("\n  Quantizing linear layers to Dynamic NVFP4...")
    fp4_state = {}  # {param_name: (packed_groups, scales, shape)}
    orig_params = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data
            packed, scales, shape = quantize_fp4(w, group_size=32)
            fp4_state[name] = (packed, scales, shape)
            orig_params[name] = w.clone()
            # Replace mod.weight with a tiny buffer (not used in forward)
            # In real Unsloth, the forward pass would be intercepted by a fused kernel
            print(f"    {name:30s} {str(list(w.shape)):18s} "
                  f"{w.numel() * 2 / 1024:.1f} KB → "
                  f"{w.numel() * 0.5 / 1024:.1f} KB (4× smaller)")

    # ── Memory accounting ──
    total_fp16 = sum(p.numel() * 2 for p in model.parameters()) / 1024
    total_quantized = sum(
        0.5 * shape[0] * shape[1]  # nibbles (0.5 bytes per value)
        + 4 * (shape[0] * shape[1] + 31) // 32  # FP32 scales
        for _, _, shape in fp4_state.values()
    ) / 1024
    print(f"\n  Total FP16 size:   {total_fp16:.1f} KB")
    print(f"  Total NVFP4 size:  {total_quantized:.1f} KB")
    print(f"  Compression ratio: {total_fp16 / total_quantized:.1f}×")

    # ── Fused dequant-matmul forward pass (simulated) ──
    print("\n  Running fused dequantize-matmul forward pass...")
    pred_nvfp4 = model.generate(prompt, max_new=30)
    print(f"  NVFP4 generate (naive, weights not actually replaced): {pred_nvfp4[:10]}...")

    # Actually replace weights with quantized version and run again
    print("\n  Actually dequantizing and replacing weights...")
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in fp4_state:
            packed, scales, shape = fp4_state[name]
            mod.weight.data = dequantize_fp4(packed, scales, shape, group_size=32)

    pred_actual = model.generate(prompt, max_new=30)
    print(f"  NVFP4 generate (dequantized weights): {pred_actual[:10]}...")

    # ── Accuracy comparison ──
    # Measure on a held-out batch
    test_x = torch.randint(0, 1000, (8, 32))
    train_orig = torch.optim.Adam(orig_params.values(), lr=0)
    # Original model forward
    model_orig = TinyLM()
    for name, p in model_orig.named_parameters():
        if name in [n + '.weight' for n, m in model_orig.named_modules() if isinstance(m, nn.Linear)]:
            mod_name = '.'.join(name.split('.')[:-1])
            if mod_name in orig_params:
                p.data = orig_params[mod_name].clone()
    logits_orig = model_orig(test_x)
    logits_quant = model(test_x)

    diff = (logits_orig - logits_quant).abs().mean().item()
    max_diff = (logits_orig - logits_quant).abs().max().item()
    acc_match = (logits_orig.argmax(-1) == logits_quant.argmax(-1)).float().mean().item()

    print(f"\n  {'─'*50}")
    print(f"  Accuracy Impact of NVFP4 Quantization")
    print(f"  {'─'*50}")
    print(f"  Mean abs logit diff:  {diff:.5f}")
    print(f"  Max abs logit diff:   {max_diff:.5f}")
    print(f"  Argmax agreement:     {acc_match*100:.1f}%")
    print(f"  {'─'*50}")

    # ── Show the FP4 value grid ──
    print(f"\n  FP4 E2M1 representable grid (16 values after dedup):")
    print(f"    {[round(v, 2) for v in FP4_QUANTIZED.float().tolist()]}")
    print(f"\n  Done. Memory saved: {total_fp16 / total_quantized:.1f}×")

if __name__ == "__main__":
    main()
