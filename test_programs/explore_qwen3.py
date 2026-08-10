"""
Learning exercise: load Qwen3-0.6B from Hugging Face, print every layer,
and trace one forward pass through the architecture manually.

Paste into one Colab cell (T4 GPU) or run locally:
    python explore_qwen3.py

Sections:
  1. Load config + tokenizer + model (meta device = no RAM cost for inspection)
  2. Print the full module tree
  3. Print per-layer parameter shapes
  4. Trace activations through one forward pass with hooks
  5. Manual layer-by-layer walk (embedding -> blocks -> norm -> lm_head)
"""

# ============================================================
# 0. Install
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch", "transformers", "accelerate", "safetensors"])

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"

# ============================================================
# 1. Config: the architecture blueprint
# ============================================================
config = AutoConfig.from_pretrained(MODEL_ID)
print("=" * 60)
print("ARCHITECTURE CONFIG")
print("=" * 60)
for k in ["model_type", "hidden_size", "intermediate_size",
          "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "head_dim",
          "max_position_embeddings", "rope_theta",
          "rms_norm_eps", "vocab_size", "tie_word_embeddings"]:
    print(f"  {k:28s} = {getattr(config, k, '(n/a)')}")

# Derived facts
n_layers = config.num_hidden_layers
n_q = config.num_attention_heads
n_kv = config.num_key_value_heads
d = config.hidden_size
print(f"\n  GQA ratio (q_heads / kv_heads) = {n_q // n_kv}")
print(f"  Total layers = {n_layers}")

# ============================================================
# 2. Module tree (meta device: structure only, zero memory)
# ============================================================
print("\n" + "=" * 60)
print("MODULE TREE")
print("=" * 60)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

def print_tree(module, prefix="", depth=0, max_depth=3):
    if depth > max_depth:
        return
    for name, child in module.named_children():
        params = sum(p.numel() for p in child.parameters(recurse=False))
        p_str = f"  [{params:,} params]" if params else ""
        print(f"{prefix}{name}: {child.__class__.__name__}{p_str}")
        print_tree(child, prefix + "  ", depth + 1, max_depth)

print_tree(model)

# ============================================================
# 3. Parameter shapes per layer
# ============================================================
print("\n" + "=" * 60)
print("PARAMETER SHAPES (layer 0 + global)")
print("=" * 60)
shown = 0
for name, p in model.named_parameters():
    if ".0." in name or "model.layers" not in name:
        print(f"  {name:60s} {str(list(p.shape)):22s} {p.numel():>12,}")
        shown += 1
print(f"\n  ... (layers 1..{n_layers-1} identical in shape to layer 0)")
total = sum(p.numel() for p in model.parameters())
print(f"\n  TOTAL PARAMETERS: {total:,}  (~{total * 2 / 1e9:.2f} GB in bf16/fp16)")

# ============================================================
# 4. Load REAL weights (GPU) and trace activations with hooks
# ============================================================
print("\n" + "=" * 60)
print("LIVE FORWARD PASS — activation shapes per layer")
print("=" * 60)
del model  # drop the meta model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="cuda"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print(f"Prompt: {prompt!r}  ->  input_ids shape {list(inputs['input_ids'].shape)}")

# Register hooks on each decoder layer
activations = []
hooks = []
def make_hook(idx):
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        activations.append((idx, tuple(h.shape)))
    return hook

for i, layer in enumerate(model.model.layers):
    hooks.append(layer.register_forward_hook(make_hook(i)))

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

for h in hooks:
    h.remove()

print(f"\nEmbedding output:        {tuple(out.hidden_states[0].shape)}")
for idx, shape in activations:
    print(f"Decoder layer {idx:2d} output: {shape}")
print(f"Final norm / logits:     {tuple(out.logits.shape)}")

next_id = out.logits[0, -1].argmax().item()
print(f"\nArgmax next token: {next_id} = {tokenizer.decode([next_id])!r}")

# ============================================================
# 5. Manual walk: run the pipeline piece by piece
# ============================================================
print("\n" + "=" * 60)
print("MANUAL WALK THROUGH THE FORWARD PASS")
print("=" * 60)
with torch.no_grad():
    ids = inputs["input_ids"]
    h = model.model.embed_tokens(ids)
    print(f"1. embed_tokens:            {list(ids.shape)} -> {list(h.shape)}")

    pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    pos_emb = model.model.rotary_emb(h, pos)
    print(f"2. rotary_emb (cos,sin):    {[tuple(t.shape) for t in pos_emb]}")

    for i, layer in enumerate(model.model.layers):
        h = layer(h, attention_mask=None, position_ids=pos,
                  position_embeddings=pos_emb)[0]
        if i < 3 or i == n_layers - 1:
            print(f"3.{i:2d} decoder layer out:     {list(h.shape)}")
        elif i == 3:
            print(f"     ... ({n_layers - 4} more identical layers)")

    h = model.model.norm(h)
    print(f"4. final rms_norm:          {list(h.shape)}")

    logits = model.lm_head(h)
    print(f"5. lm_head -> logits:       {list(logits.shape)}")
    print(f"\n   sanity: logits argmax == {logits[0, -1].argmax().item()}")

print("\nDone. Try editing `prompt` or inspecting layer.self_attn.q_proj.weight!")
