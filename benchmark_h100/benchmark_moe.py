#!/usr/bin/env python3
"""Run vLLM's random serving benchmark across a small model/workload matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODELS = [
    "Qwen/Qwen3.5-35B-A3B-FP8",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
]


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def wait_for_server(base_url: str, process: subprocess.Popen[str], timeout: int) -> None:
    deadline = time.time() + timeout
    url = base_url.rstrip("/") + "/v1/models"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM did not become ready within {timeout}s; inspect the server log")


def print_kv_cache_capacity(log_path: Path) -> None:
    """Print capacity values emitted by vLLM after cache initialization."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    token_match = re.search(r"GPU KV cache size:\s*([\d,]+) tokens", text)
    concurrency_match = re.search(
        r"Maximum concurrency for\s*([\d,]+) tokens per request:\s*([\d.]+)x", text
    )
    print(
        "KV CACHE CAPACITY:",
        f"{token_match.group(1)} tokens" if token_match else "not found in this vLLM log format",
        flush=True,
    )
    if concurrency_match:
        print(
            "KV CACHE MAX CONCURRENCY:",
            f"{concurrency_match.group(2)}x at {concurrency_match.group(1)} tokens/request",
            flush=True,
        )


def prometheus_value(text: str, name: str) -> float:
    values = []
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                values.append(float(line.rsplit(maxsplit=1)[-1]))
            except ValueError:
                pass
    return sum(values)


class MetricsSampler:
    """Sample useful live vLLM gauges while a benchmark is running."""

    def __init__(self, base_url: str) -> None:
        self.url = base_url.rstrip("/") + "/metrics"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running: list[float] = []
        self.waiting: list[float] = []
        self.kv_usage: list[float] = []

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=3) as response:
                    body = response.read().decode("utf-8", errors="replace")
                self.running.append(prometheus_value(body, "vllm:num_requests_running"))
                self.waiting.append(prometheus_value(body, "vllm:num_requests_waiting"))
                self.kv_usage.append(prometheus_value(body, "vllm:kv_cache_usage_perc"))
            except (urllib.error.URLError, TimeoutError):
                pass
            self.stop_event.wait(0.5)

    def __enter__(self) -> "MetricsSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def summary(self) -> dict[str, float]:
        return {
            "mean_running_requests": statistics.fmean(self.running) if self.running else 0,
            "peak_running_requests": max(self.running, default=0),
            "peak_waiting_requests": max(self.waiting, default=0),
            "peak_kv_cache_usage_pct": 100 * max(self.kv_usage, default=0),
        }


def metric(result: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in result:
            return result[name]
    return ""


def mean_or_blank(values: list[float]) -> float | str:
    return statistics.fmean(values) if values else ""


def median_or_blank(values: list[float]) -> float | str:
    return statistics.median(values) if values else ""


def normalized_row(
    result: dict[str, Any], model: str, gpu: str, input_len: int,
    output_len: int, request_rate: float, max_num_seqs: int,
    max_num_batched_tokens: int, live: dict[str, float],
) -> dict[str, Any]:
    input_lens = [float(x) for x in result.get("input_lens", [])]
    output_lens = [float(x) for x in result.get("output_lens", [])]
    errors = result.get("errors", [])
    achieved_rps = metric(result, "request_throughput")
    p95_ttft_ms = metric(result, "p95_ttft_ms")
    throughput_ratio = (
        float(achieved_rps) / request_rate
        if achieved_rps not in ("", None) and request_rate > 0
        else ""
    )
    ttft_sla_pass = (
        float(p95_ttft_ms) <= 800
        if p95_ttft_ms not in ("", None)
        else False
    )
    throughput_sla_pass = (
        float(throughput_ratio) >= 0.95
        if throughput_ratio not in ("", None)
        else False
    )
    return {
        "gpu": gpu,
        "model": model,
        "target_input_tokens": input_len,
        "target_output_tokens": output_len,
        "offered_rps": request_rate,
        "configured_max_num_seqs": max_num_seqs,
        "configured_max_num_batched_tokens": max_num_batched_tokens,
        **live,
        "completed": metric(result, "completed"),
        "duration_s": metric(result, "duration"),
        "request_throughput_rps": achieved_rps,
        "throughput_ratio": throughput_ratio,
        "ttft_sla_pass": ttft_sla_pass,
        "throughput_sla_pass": throughput_sla_pass,
        "performance_sla_pass": ttft_sla_pass and throughput_sla_pass,
        "input_throughput_tps": metric(result, "input_throughput"),
        "output_throughput_tps": metric(result, "output_throughput"),
        "total_token_throughput_tps": metric(result, "total_token_throughput"),
        "mean_input_tokens": mean_or_blank(input_lens),
        "median_input_tokens": median_or_blank(input_lens),
        "mean_output_tokens": mean_or_blank(output_lens),
        "median_output_tokens": median_or_blank(output_lens),
        "mean_ttft_ms": metric(result, "mean_ttft_ms"),
        "p50_ttft_ms": metric(result, "p50_ttft_ms", "median_ttft_ms"),
        "p95_ttft_ms": p95_ttft_ms,
        "mean_tpot_ms": metric(result, "mean_tpot_ms"),
        "p50_tpot_ms": metric(result, "p50_tpot_ms", "median_tpot_ms"),
        "p95_tpot_ms": metric(result, "p95_tpot_ms"),
        "mean_itl_ms": metric(result, "mean_itl_ms"),
        "p50_itl_ms": metric(result, "p50_itl_ms", "median_itl_ms"),
        "p95_itl_ms": metric(result, "p95_itl_ms"),
        "mean_e2el_ms": metric(result, "mean_e2el_ms"),
        "p95_e2el_ms": metric(result, "p95_e2el_ms"),
        "error_count": len(errors) if isinstance(errors, list) else "",
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--gpu-label", default="H100-PCIe-80GB")
    parser.add_argument("--input-lengths", nargs="+", type=int, default=[3000, 8000])
    parser.add_argument("--output-length", type=int, default=300)
    parser.add_argument("--request-rates", nargs="+", type=float,
                        default=[1.157, 3.0, 5.0, 10.0])
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=64,
                        help="Server-side maximum sequences in a scheduler batch")
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768,
                        help="Server-side maximum tokens processed per scheduler iteration")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--language-model-only-patterns", nargs="*", default=["Qwen3.5"],
        help=("Append vLLM's --language-model-only flag when any substring "
              "matches the model name or local model path"),
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{args.port}"
    rows: list[dict[str, Any]] = []

    for model in args.models:
        model_dir = args.results_dir / slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        log_path = model_dir / "vllm-server.log"
        server_cmd = [
            "vllm", "serve", model,
            "--dtype", args.dtype,
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--port", str(args.port),
        ]
        if any(pattern in model for pattern in args.language_model_only_patterns):
            server_cmd.append("--language-model-only")
        print("SERVER:", " ".join(server_cmd), flush=True)
        if args.dry_run:
            server = None
            log_handle = None
        else:
            log_handle = log_path.open("w", encoding="utf-8")
            server = subprocess.Popen(
                server_cmd, stdout=log_handle, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            wait_for_server(base_url, server, args.startup_timeout)
            print_kv_cache_capacity(log_path)
            print(
                "BATCH LIMITS:",
                f"max_num_seqs={args.max_num_seqs},",
                f"max_num_batched_tokens={args.max_num_batched_tokens},",
                f"client_max_concurrency={args.max_concurrency}",
                flush=True,
            )

        try:
            for input_len in args.input_lengths:
                for request_rate in args.request_rates:
                    stem = f"in{input_len}-out{args.output_length}-rps{request_rate:g}"
                    result_path = model_dir / f"{stem}.json"
                    bench_cmd = [
                        "vllm", "bench", "serve",
                        "--backend", "vllm",
                        "--base-url", base_url,
                        "--endpoint", "/v1/completions",
                        "--model", model,
                        "--dataset-name", "random",
                        "--random-input-len", str(input_len),
                        "--random-output-len", str(args.output_length),
                        "--random-range-ratio", "0.0",
                        "--request-rate", str(request_rate),
                        "--num-prompts", str(args.num_prompts),
                        "--max-concurrency", str(args.max_concurrency),
                        "--ignore-eos",
                        "--percentile-metrics", "ttft,tpot,itl,e2el",
                        "--metric-percentiles", "50,95,99",
                        "--save-result",
                        "--save-detailed",
                        "--result-dir", str(model_dir),
                        "--result-filename", result_path.name,
                        "--metadata",
                        f"gpu={args.gpu_label}",
                        f"target_input_tokens={input_len}",
                        f"target_output_tokens={args.output_length}",
                        f"offered_rps={request_rate}",
                    ]
                    print("BENCH:", " ".join(bench_cmd), flush=True)
                    if args.dry_run:
                        continue
                    with MetricsSampler(base_url) as sampler:
                        subprocess.run(bench_cmd, check=True)
                    live = sampler.summary()
                    print(
                        "OBSERVED:",
                        f"peak_running_requests={live['peak_running_requests']:.0f},",
                        f"mean_running_requests={live['mean_running_requests']:.1f},",
                        f"peak_waiting_requests={live['peak_waiting_requests']:.0f},",
                        f"peak_kv_cache_usage={live['peak_kv_cache_usage_pct']:.1f}%",
                        flush=True,
                    )
                    with result_path.open(encoding="utf-8") as handle:
                        result = json.load(handle)
                    rows.append(normalized_row(
                        result, model, args.gpu_label, input_len,
                        args.output_length, request_rate, args.max_num_seqs,
                        args.max_num_batched_tokens, live,
                    ))
                    write_summary(args.results_dir / "summary.csv", rows)
        finally:
            if server is not None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            if log_handle is not None:
                log_handle.close()

    if not args.dry_run:
        print(f"Wrote {args.results_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

