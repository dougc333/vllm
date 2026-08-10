"""
Minimal PyTorch shared memory — no manual control plane, no manual ring buffers.
PyTorch handles all metadata (shape, dtype, strides) automatically.

Key: torch.multiprocessing.Queue transparently puts tensors into shared memory.
The data never gets pickled — only a small metadata handle crosses the queue.
"""
import torch
import torch.multiprocessing as mp

DTYPE = torch.float32
ELEMS = 4096
N_TENSORS = 2

def worker(q: mp.Queue, ready: mp.Event, done: mp.Event):
    """Receive tensor metadata handles from queue, read data from shared memory."""
    ready.wait()
    t0 = q.get()  # no copy — just a handle into existing shared memory
    t1 = q.get()  # same

    print(f"Worker: t0 mean={t0.mean().item():.1f}, "
          f"t1[0]={t1[0].item()}, t1[-1]={t1[-1].item()}")

    assert (t0 == 42.0).all()
    assert (t1 == torch.arange(ELEMS, dtype=DTYPE)).all()
    print("Worker: verification passed")
    done.set()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    q = mp.Queue()
    ready = mp.Event()
    done = mp.Event()

    p = mp.Process(target=worker, args=(q, ready, done))
    p.start()

    # Create tensors and put in shared memory.
    # share_memory_() tells PyTorch: "put this tensor where other procs can see it."
    t0 = torch.full((ELEMS,), 42.0, dtype=DTYPE).share_memory_()
    t1 = torch.arange(ELEMS, dtype=DTYPE).share_memory_()

    # Queue.send only sends metadata (storage handle + shape + dtype).
    # The 32 KB of float data never moves — it stays in shared memory.
    q.put(t0)
    q.put(t1)
    ready.set()

    done.wait()
    p.join()
    print("Main: done")
