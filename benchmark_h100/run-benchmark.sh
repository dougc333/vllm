#!/usr/bin/env bash
set -euo pipefail

python benchmark_random.py \
  --models \
    Qwen/Qwen3-4B \
    Qwen/Qwen3-8B \
    Qwen/Qwen3-14B \
  --gpu-label A100-80GB \
  --input-lengths 3000 8000 \
  --output-length 300 \
  --request-rates 1.157 10 \
  --num-prompts 200 \
  --max-concurrency 64 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 32768 \
  --max-model-len 16384 \
  --results-dir results_h100
