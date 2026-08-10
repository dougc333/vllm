"""
Single-cell Colab: vLLM benchmark Qwen3-0.6B TTFT p95/p99.
Copy-paste into one Colab cell (T4 GPU, ~5 min runtime).
"""
# ── 1. Install ─────────────────────────────────────────
!pip install -q vllm datasets

import subprocess, time, json, os, signal
import numpy as np

# ── 2. Config ──────────────────────────────────────────
MODEL = "Qwen/Qwen3-0.6B"
PORT = 8000
MODEL_LEN = 2048

# ── 3. Start vLLM server in background ─────────────────
print(f"Starting vLLM server with {MODEL}...")
server = subprocess.Popen(
    ["python", "-m", "vllm.entrypoints.openai.api_server",
     "--model", MODEL,
     "--max-model-len", str(MODEL_LEN),
     "--gpu-memory-utilization", "0.85",
     "--port", str(PORT)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True,
)

# Wait for server to be ready
for attempt in range(60):
    time.sleep(5)
    try:
        import urllib.request
        r = urllib.request.urlopen(f"http://localhost:{PORT}/health")
        if r.status == 200:
            print(f"Server ready after {(attempt+1)*5}s")
            break
    except Exception:
        pass
    if attempt % 6 == 5:
        print(f"  waiting... {((attempt+1)*5)}s")
else:
    raise RuntimeError("Server did not start")

# ── 4. Run benchmark via vLLM CLI ──────────────────────
print("\nRunning benchmark...")
bench = subprocess.run(
    ["python", "-m", "vllm.entrypoints.cli.main", "bench", "serve",
     "--backend", "openai",
     "--model", MODEL,
     "--endpoint", f"/v1/completions",
     "--base-url", f"http://localhost:{PORT}",
     "--tokenizer", MODEL,
     "--request-rate", "inf",          # send all at once
     "--num-prompts", "100",
     "--dataset-name", "random",
     "--random-input-len", "512",
     "--random-output-len", "128",
     "--result-dir", "/tmp/vllm_bench",
     "--save-result",
     ],
    capture_output=True, text=True, timeout=300,
)
print(bench.stdout[-2000:] if len(bench.stdout) > 2000 else bench.stdout)

# ── 5. Parse & print results ───────────────────────────
result_path = "/tmp/vllm_bench"  # vLLM writes results here
json_files = [f for f in os.listdir(result_path) if f.endswith(".json")] if os.path.exists(result_path) else []
if json_files:
    with open(os.path.join(result_path, json_files[0])) as f:
        data = json.load(f)
    tft = data.get("ttft_mean_ms", 0)
    tft_p99 = data.get("ttft_p99_ms", 0)
    tpot = data.get("tpot_mean_ms", 0)
    thru = data.get("request_throughput", 0)
    itl = data.get("itl_mean_ms", 0)

    print("\n" + "="*55)
    print(f"  Qwen3-0.6B — vLLM (FP16, T4)")
    print("="*55)
    print(f"  TTFT mean: {tft:.1f} ms")
    print(f"  TTFT p99:  {tft_p99:.1f} ms")
    print(f"  TTFT p95:  {data.get('ttft_p95_ms', 'N/A')} ms")
    print(f"  ITL mean:  {itl:.1f} ms/token")
    print(f"  TPOT mean: {tpot:.1f} ms/token")
    print(f"  Throughput: {thru:.1f} req/s" if thru > 0 else f"  Throughput: N/A")
    print("="*55)
else:
    print("Could not find result JSON, printing raw output above.")

# ── 6. Stop server ─────────────────────────────────────
server.send_signal(signal.SIGINT)
server.wait(timeout=30)
print("\nServer stopped.")