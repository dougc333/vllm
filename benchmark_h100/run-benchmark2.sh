#!/usr/bin/env bash
set -euo pipefail

# Keep benchmark_moe.py beside this script. The runner can then be launched
# from any working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_moe.py"

if [[ ! -f "${BENCHMARK_SCRIPT}" ]]; then
  echo "ERROR: ${BENCHMARK_SCRIPT} was not found." >&2
  echo "Place benchmark_moe.py in the same directory as this script." >&2
  exit 1
fi

# Prefer the H100 virtual environment used during setup. Override with:
#   PYTHON_BIN=/path/to/python MODEL_ROOT=/path/to/models ./run-benchmark2.sh
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${HOME}/.venv312/bin/python" ]]; then
    PYTHON_BIN="${HOME}/.venv312/bin/python"
    export PATH="${HOME}/.venv312/bin:${PATH}"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

command -v vllm >/dev/null || {
  echo "ERROR: vllm is not on PATH. Activate the vLLM virtual environment." >&2
  exit 1
}

MODEL_ROOT="${MODEL_ROOT:-${HOME}/models}"

# Prefer complete local downloads, while retaining Hugging Face IDs as a fallback.
model_ref() {
  local local_name="$1"
  local hub_id="$2"
  if [[ -f "${MODEL_ROOT}/${local_name}/config.json" ]]; then
    printf '%s\n' "${MODEL_ROOT}/${local_name}"
  else
    printf '%s\n' "${hub_id}"
  fi
}

MODELS=(
  "$(model_ref Qwen3.5-35B-A3B-FP8 Qwen/Qwen3.5-35B-A3B-FP8)"
  "$(model_ref gpt-oss-20b openai/gpt-oss-20b)"
  "$(model_ref Qwen3-30B-A3B-Instruct-2507-FP8 Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)"
)

"${PYTHON_BIN}" "${BENCHMARK_SCRIPT}" \
  --models "${MODELS[@]}" \
  --gpu-label H100-PCIe-80GB \
  --input-lengths 3000 8000 \
  --output-length 300 \
  --request-rates 1.157 3 5 10 \
  --num-prompts 500 \
  --max-concurrency 256 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 32768 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --dtype auto \
  --startup-timeout 1800 \
  --results-dir "${SCRIPT_DIR}/results_h100_moe"

