"""
Minimal POSIX shared memory transfer of 2 × 4096 PyTorch tensors between processes.
Zero-copy — tensors point directly into /dev/shm.

Run:  python shm_tensor_transfer.py
"""
import torch
import multiprocessing as mp
from multiprocessing import shared_memory

DTYPE = torch.float32         # 4 bytes per element
N_TENSORS = 2
ELEMS_PER_TENSOR = 4096
BYTES_PER_TENSOR = ELEMS_PER_TENSOR * DTYPE.itemsize  # 16384 bytes
TOTAL_BYTES = N_TENSORS * BYTES_PER_TENSOR             # 32768 bytes

def writer(shm_name: str, ready: mp.Event, done: mp.Event):
    """Create 2 tensors in shared memory, write data, signal reader, wait."""
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        # Map two tensors into the same shared memory block
        t0 = torch.frombuffer(shm.buf[0:BYTES_PER_TENSOR], dtype=DTYPE).reshape(ELEMS_PER_TENSOR)
        t1 = torch.frombuffer(shm.buf[BYTES_PER_TENSOR:TOTAL_BYTES], dtype=DTYPE).reshape(ELEMS_PER_TENSOR)

        # Write data
        t0[:] = torch.ones(ELEMS_PER_TENSOR) * 42.0
        t1[:] = torch.arange(ELEMS_PER_TENSOR, dtype=DTYPE)

        print(f"Writer: wrote tensor0 mean={t0.mean().item()}, "
              f"tensor1[0]={t1[0].item()}, tensor1[-1]={t1[-1].item()}")
        ready.set()  # signal reader

        done.wait()  # wait for reader to finish
        print("Writer: reader done, exiting")
    finally:
        shm.close()

def reader(shm_name: str, ready: mp.Event, done: mp.Event):
    """Wait for writer, then read tensors directly from shared memory (zero-copy)."""
    ready.wait()  # wait for writer
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        # Zero-copy: tensors point directly into the shm buffer
        t0 = torch.frombuffer(shm.buf[0:BYTES_PER_TENSOR], dtype=DTYPE).reshape(ELEMS_PER_TENSOR)
        t1 = torch.frombuffer(shm.buf[BYTES_PER_TENSOR:TOTAL_BYTES], dtype=DTYPE).reshape(ELEMS_PER_TENSOR)

        print(f"Reader: read tensor0 mean={t0.mean().item()}, "
              f"tensor1[0]={t1[0].item()}, tensor1[-1]={t1[-1].item()}")

        # Verify correctness
        assert (t0 == 42.0).all(), "tensor0 should be all 42.0"
        assert (t1 == torch.arange(ELEMS_PER_TENSOR, dtype=DTYPE)).all(), \
            "tensor1 should be 0..4095"
        print("Reader: verification passed")
        done.set()  # signal writer
    finally:
        shm.close()


if __name__ == "__main__":
    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=TOTAL_BYTES)
    name = shm.name
    shm.close()  # processes will open it independently

    print(f"Created shared memory '{name}' ({TOTAL_BYTES} bytes)")

    ready = mp.Event()
    done = mp.Event()

    p_writer = mp.Process(target=writer, args=(name, ready, done))
    p_reader = mp.Process(target=reader, args=(name, ready, done))

    p_writer.start()
    p_reader.start()

    p_writer.join()
    p_reader.join()

    # Clean up shared memory
    shm = shared_memory.SharedMemory(name=name)
    shm.unlink()
    print("Shared memory cleaned up")
