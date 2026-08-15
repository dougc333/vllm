#!/usr/bin/env bash
set -Eeuo pipefail

# Single-file H100 benchmark for the customer-support SLA:
#   median input 3,000 tokens; p95 input 8,000 tokens; output 300 tokens
#   average 1.157 RPS; peak 10 RPS; p95 TTFT <= 800 ms

if [[ -x "${HOME}/.venv312/bin/python" ]]; then
  export PATH="${HOME}/.venv312/bin:${PATH}"
  PYTHON_BIN="${PYTHON_BIN:-${HOME}/.venv312/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
fi

for required_command in vllm curl setsid; do
  command -v "${required_command}" >/dev/null || {
    echo "ERROR: ${required_command} is not available." >&2
    exit 1
  }
done

MODEL_ROOT="${MODEL_ROOT:-${HOME}/models}"
RESULTS_ROOT="${RESULTS_ROOT:-${HOME}/results_h100_moe_better}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_ID}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"

INPUT_LENGTHS=(3000 8000)
REQUEST_RATES=(1.157 3 5 10)
OUTPUT_LENGTH=300

mkdir -p "${RESULTS_DIR}"
export RESULTS_DIR

model_ref() {
  local local_name="$1"
  local hub_id="$2"
  if [[ -f "${MODEL_ROOT}/${local_name}/config.json" ]]; then
    printf '%s\n' "${MODEL_ROOT}/${local_name}"
  else
    printf '%s\n' "${hub_id}"
  fi
}

slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's|[^a-z0-9_.-]+|-|g; s/^-+//; s/-+$//'
}

# Run the smallest model first so useful results arrive quickly.
MODELS=(
  "$(model_ref Qwen3-30B-A3B-Instruct-2507-FP8 Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)"
  "$(model_ref Qwen3.5-35B-A3B-FP8 Qwen/Qwen3.5-35B-A3B-FP8)"
)

# Some vLLM releases can skip loading the multimodal tower for text-only runs.
# Detect the option instead of assuming it exists; vLLM 0.27.1 does not expose
# --language-model-only.
VLLM_SERVE_HELP="$(vllm serve --help 2>&1 || true)"
LANGUAGE_MODEL_ONLY_SUPPORTED=0
if [[ "${VLLM_SERVE_HELP}" == *"--language-model-only"* ]]; then
  LANGUAGE_MODEL_ONLY_SUPPORTED=1
fi
unset VLLM_SERVE_HELP

SERVER_PID=""

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

on_exit() {
  stop_server
}
trap on_exit EXIT INT TERM

wait_for_server() {
  local log_path="$1"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "ERROR: vLLM exited during startup. Last 100 log lines:" >&2
      tail -n 100 "${log_path}" >&2 || true
      return 1
    fi
    if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 3
  done
  echo "ERROR: vLLM was not ready after ${STARTUP_TIMEOUT}s. See ${log_path}" >&2
  return 1
}

if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: ${BASE_URL} already has a running server. Stop it or set PORT." >&2
  exit 1
fi

echo "Results: ${RESULTS_DIR}"
echo "Server limits: max_num_seqs=${MAX_NUM_SEQS}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "Client: max_concurrency=${MAX_CONCURRENCY}, prompts_per_run=${NUM_PROMPTS}"

for model in "${MODELS[@]}"; do
  model_slug="$(slug "${model}")"
  model_dir="${RESULTS_DIR}/${model_slug}"
  log_path="${model_dir}/vllm-server.log"
  mkdir -p "${model_dir}"

  server_cmd=(
    vllm serve "${model}"
    --dtype auto
    # Avoid DeepGEMM JIT on this host: its nvcc is older than the >=12.3
    # required by the bundled DeepGEMM compiler. Triton supports these
    # block-scaled FP8 checkpoints and does not depend on that nvcc path.
    --linear-backend triton
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --port "${PORT}"
  )

  # This benchmark is text-only; do not load Qwen3.5's vision tower.
  if [[ "${model}" == *Qwen3.5* ]] && (( LANGUAGE_MODEL_ONLY_SUPPORTED )); then
    server_cmd+=(--language-model-only)
  fi

  echo
  echo "SERVER: ${server_cmd[*]}"
  setsid "${server_cmd[@]}" >"${log_path}" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server "${log_path}"; then
    stop_server
    echo "SKIP: ${model} could not start."
    continue
  fi

  grep -E "GPU KV cache size|Maximum concurrency" "${log_path}" || true

  for input_len in "${INPUT_LENGTHS[@]}"; do
    for request_rate in "${REQUEST_RATES[@]}"; do
      result_name="in${input_len}-out${OUTPUT_LENGTH}-rps${request_rate}.json"

      bench_cmd=(
        vllm bench serve
        --backend openai-chat
        --base-url "${BASE_URL}"
        --endpoint /v1/chat/completions
        --model "${model}"
        --dataset-name random
        --random-input-len "${input_len}"
        --random-output-len "${OUTPUT_LENGTH}"
        --random-range-ratio 0.0
        --request-rate "${request_rate}"
        --num-prompts "${NUM_PROMPTS}"
        --max-concurrency "${MAX_CONCURRENCY}"
        --num-warmups 10
        --temperature 0
        --ignore-eos
        --extra-body '{"min_tokens":300,"ignore_eos":true}'
        --goodput ttft:800
        --percentile-metrics ttft,tpot,itl,e2el
        --metric-percentiles 50,95,99
        --save-result
        --save-detailed
        --result-dir "${model_dir}"
        --result-filename "${result_name}"
        --metadata
        gpu=H100-PCIe-80GB
        "target_input_tokens=${input_len}"
        "target_output_tokens=${OUTPUT_LENGTH}"
        "offered_rps=${request_rate}"
        ttft_sla_ms=800
      )

      echo
      echo "BENCH: model=${model}, input=${input_len}, output=${OUTPUT_LENGTH}, offered_rps=${request_rate}"
      if ! "${bench_cmd[@]}"; then
        echo "FAILED: ${model}, input=${input_len}, rate=${request_rate}" >&2
      fi
    done
  done

  stop_server
  sleep 3
done

# Build a compact CSV from every successful result. This intentionally accepts
# several vLLM result-key spellings so it remains useful across minor releases.
"${PYTHON_BIN}" <<'PY'
import csv
import json
import os
import re
import statistics
from pathlib import Path

root = Path(os.environ["RESULTS_DIR"])
rows = []
pattern = re.compile(r"in(?P<input>\d+)-out(?P<output>\d+)-rps(?P<rate>[\d.]+)\.json$")

def first(data, *keys):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return ""

for path in sorted(root.glob("*/*.json")):
    match = pattern.match(path.name)
    if not match:
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue

    offered = float(match.group("rate"))
    achieved = first(data, "request_throughput")
    p95_ttft = first(data, "p95_ttft_ms")
    goodput = first(data, "request_goodput", "goodput")
    input_lens = [float(value) for value in data.get("input_lens", [])]
    output_lens = [float(value) for value in data.get("output_lens", [])]
    throughput_ratio = float(achieved) / offered if achieved != "" else ""
    ttft_pass = p95_ttft != "" and float(p95_ttft) <= 800.0
    throughput_pass = throughput_ratio != "" and throughput_ratio >= 0.95
    goodput_pass = goodput == "" or float(goodput) >= 0.95 * offered

    rows.append({
        "model": first(data, "model_id", "model") or path.parent.name,
        "target_input_tokens": int(match.group("input")),
        "target_output_tokens": int(match.group("output")),
        "offered_rps": offered,
        "request_throughput_rps": achieved,
        "output_throughput_tps": first(data, "output_throughput"),
        "total_generated_tokens": first(data, "total_output_tokens", "total_generated_tokens"),
        "throughput_ratio": throughput_ratio,
        "request_goodput_rps_ttft_le_800ms": goodput,
        "median_input_tokens": statistics.median(input_lens) if input_lens else "",
        "mean_output_tokens": statistics.fmean(output_lens) if output_lens else "",
        "median_output_tokens": statistics.median(output_lens) if output_lens else "",
        "p50_ttft_ms": first(data, "p50_ttft_ms", "median_ttft_ms"),
        "p95_ttft_ms": p95_ttft,
        "p99_ttft_ms": first(data, "p99_ttft_ms"),
        "p95_tpot_ms": first(data, "p95_tpot_ms"),
        "p95_itl_ms": first(data, "p95_itl_ms"),
        "p95_e2el_ms": first(data, "p95_e2el_ms"),
        "ttft_sla_pass": ttft_pass,
        "throughput_sla_pass": throughput_pass,
        "performance_sla_pass": ttft_pass and throughput_pass and goodput_pass,
    })

summary = root / "summary.csv"
if rows:
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary}")
else:
    print("No successful JSON results were found; summary.csv was not written.")
PY

echo "Complete: ${RESULTS_DIR}"
