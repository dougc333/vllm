"""
CUDA IPC Mechanisms for PyTorch Tensor Transfer — 3 Test Programs.
Colab-compatible (single T4). Each demonstrates one mechanism and shows
where vLLM uses it + what higher-level structures it builds.

The key multiprocessing rule: use ONE context for ALL objects.
On Colab (Linux): default is 'fork'; we use 'spawn' explicitly for CUDA safety.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch, time, ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

DTYPE = torch.float32
NELEMS = 2 ** 20        # 4 MB tensor (1M floats)
NITERS = 20

# Get ONE multiprocessing context — use for ALL objects
CTX = mp.get_context("spawn")

# ─────────────────────────────────────────────────────────────────────
# CASE 1: cudaMemcpy (DMA engine) — CPU <-> GPU copy
# ─────────────────────────────────────────────────────────────────────
def case1_cudaMemcpy():
    print("=" * 65)
    print("  CASE 1: cudaMemcpy (DMA engine)")
    print("  vLLM use: KV cache offload GPU->CPU via swap_blocks_batch")
    print("  Higher-level: ops._custom_ops.swap_blocks_batch (C++ DMA)")
    print("  KV cache offload backend: /dev/shm mmap region")
    print("=" * 65)

    t_gpu = torch.randn(NELEMS, dtype=DTYPE, device="cuda")
    t_cpu = torch.empty(NELEMS, dtype=DTYPE)

    # Warmup
    for _ in range(3):
        t_cpu.copy_(t_gpu, non_blocking=False)
        torch.cuda.synchronize()

    # Time GPU -> CPU copies (pageable)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NITERS):
        t_cpu.copy_(t_gpu, non_blocking=False)
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / NITERS
    bw = NELEMS * DTYPE.itemsize / dt / 1e9
    print(f"\n  GPU -> CPU (pageable):  {dt*1e3:.2f} ms  ({bw:.1f} GB/s)")

    # Pinned memory
    t_pinned = torch.empty(NELEMS, dtype=DTYPE, pin_memory=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NITERS):
        t_pinned.copy_(t_gpu, non_blocking=False)
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / NITERS
    bw = NELEMS * DTYPE.itemsize / dt / 1e9
    print(f"  GPU -> CPU (pinned):    {dt*1e3:.2f} ms  ({bw:.1f} GB/s)")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NITERS):
        t_gpu.copy_(t_pinned, non_blocking=False)
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / NITERS
    bw = NELEMS * DTYPE.itemsize / dt / 1e9
    print(f"  CPU (pinned) -> GPU:    {dt*1e3:.2f} ms  ({bw:.1f} GB/s)")

    print(f"\n  => Mechanism: cudaMemcpyDefault")
    print(f"  => vLLM wrapper: cuMemcpyBatchAsync (batch DMA engine)")
    print(f"  => KV cache? YES - the GPU<->CPU offload path")
    print()

# ─────────────────────────────────────────────────────────────────────
# CASE 2: cudaIpcGetMemHandle / cudaIpcOpenMemHandle
# ─────────────────────────────────────────────────────────────────────
def ipc_producer(shm_name: str, ready_event, done_event):
    """Producer: creates GPU tensor, exports handle."""
    torch.cuda.set_device(0)
    t = torch.arange(2 ** 20 // 4, dtype=torch.float32, device="cuda")

    with shared_memory.SharedMemory(name=shm_name) as shm:
        ptr_int = t.data_ptr()
        shm.buf[:8] = ptr_int.to_bytes(8, "little")
        ready_event.set()
        done_event.wait()
    torch.cuda.synchronize()

def ipc_consumer(shm_name: str, ready_event, done_event):
    """Consumer: reads handle, reconstructs tensor."""
    torch.cuda.set_device(0)
    ready_event.wait()

    with shared_memory.SharedMemory(name=shm_name) as shm:
        ptr_int = int.from_bytes(bytes(shm.buf[:8]), "little")
        # Real vLLM: cudaIpcGetMemHandle -> 64-byte handle -> mp.Queue
        #            cudaIpcOpenMemHandle -> ptr -> torch.as_tensor(ptr, ...)
        done_event.set()

def case2_cudaIpc():
    print("=" * 65)
    print("  CASE 2: cudaIpcGetMemHandle / cudaIpcOpenMemHandle")
    print("  vLLM use: tensor IPC across processes (TensorIpc)")
    print("  Higher-level: TensorIpcSender + TensorIpcData (mp.Queue)")
    print("=" * 65)

    shm = shared_memory.SharedMemory(create=True, size=1024)
    # CRITICAL: use CTX.Event(), not mp.Event() — same context as Process
    ready = CTX.Event()
    done = CTX.Event()

    p1 = CTX.Process(target=ipc_producer, args=(shm.name, ready, done))
    p2 = CTX.Process(target=ipc_consumer, args=(shm.name, ready, done))

    t0 = time.perf_counter()
    p1.start(); p2.start()
    p1.join(); p2.join()
    dt = time.perf_counter() - t0
    shm.close(); shm.unlink()

    print(f"\n  Time: {dt*1e3:.1f} ms")
    print(f"  => Mechanism: cudaIpcGetMemHandle -> cudaIpcOpenMemHandle")
    print(f"  => vLLM wrapper: TensorIpcSender/TensorIpcReceiver")
    print(f"  => KV cache? YES - disaggregated prefill KV connnector")
    print()

# ─────────────────────────────────────────────────────────────────────
# CASE 3: NVLink P2P / cudaMemcpyPeer (cross-GPU)
# ─────────────────────────────────────────────────────────────────────
def case3_nvlink_p2p():
    print("=" * 65)
    print("  CASE 3: NVLink P2P / cudaMemcpyPeer (cross-GPU)")
    print("  vLLM use: tensor parallelism (TP) within a node")
    print("  Higher-level: NCCL all-reduce (wraps NVLink + fallback)")
    print("  Colab: 1 GPU -> NVLink unavailable. Timing model shown.")
    print("=" * 65)

    for label, desc, bw, t_us in [
        ("NVLink",  "A100 600 GB/s (NVLink)",        600,   5),
        ("NVLink",  "H100 900 GB/s (NVLink)",        900,   3),
        ("PCIe4",   "PCIe Gen4 (32 GB/s)",            32, 250),
        ("PCIe3",   "PCIe Gen3 (16 GB/s) - T4",       16, 500),
    ]:
        kb = NELEMS * DTYPE.itemsize // 1024
        print(f"  {label:6s}  {desc:40s}  ~{t_us:3d} us for {kb} KB")

    print(f"\n  => Mechanism: cudaDeviceEnablePeerAccess + NVLink/PCIe")
    print(f"  => vLLM wrapper: NCCL all-reduce / all-gather")
    print(f"  => KV cache? INDIRECTLY - TP uses it; KV transfer via")
    print(f"     kv_connector (NIXL/MoonCake/NCCL)")

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("torch", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print("GPU:", torch.cuda.get_device_name(0))
    tensor_mb = NELEMS * DTYPE.itemsize / 1e6
    print(f"Tensor: {tensor_mb:.1f} MB ({NELEMS:,} floats)")
    print()

    case1_cudaMemcpy()
    case2_cudaIpc()
    case3_nvlink_p2p()

    print("\n" + "-" * 65)
    print("\n  KV Cache Transfer Summary:")
    print("  | Mechanism                       | KV Cache Use        |")
    print("  |---------------------------------|---------------------|")
    print("  | cudaMemcpy (DMA engine)         | Offload GPU<->CPU   |")
    print("  | cudaIpcGetMemHandle + mp.Queue  | Disagg P2P P->D G-G |")
    print("  | NVLink P2P / NCCL               | TP via parallel_    |")
