# in shared mem we saw we dont have the type info just raw bytes. 
#zeromq is used to send the type info and shape info to the other process so it can reconstruct the tensor from the shared memory.
#zeromq is called teh control channel


# ZeroMQ control channel + POSIX shared memory for zero-copy tensor IPC.
# This is the pattern used by LMCache, vLLM, and similar systems.

import torch
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import zmq
import json
import os
import math
import time


def sender_process(endpoint: str, ready_event):
    """
    SENDER: Creates a tensor, stores it in /dev/shm,
    then sends the metadata over ZeroMQ.
    """
    pid = os.getpid()
    shape = (4, 1024)
    dtype = torch.float32

    element_size = torch.empty(0, dtype=dtype).element_size()
    num_elements = math.prod(shape)
    byte_size = num_elements * element_size

    # --- 1. Allocate shared memory (creates file in /dev/shm) ---
    shm = shared_memory.SharedMemory(create=True, size=byte_size)
    tensor = torch.frombuffer(shm.buf[:byte_size], dtype=dtype).reshape(shape)
    tensor[:] = torch.arange(num_elements, dtype=dtype).reshape(shape)

    print(f"[Sender PID {pid}] Created shm: '{shm.name}', shape={shape}")
    print(f"[Sender PID {pid}] First 5 values: {tensor[0, :5]}")

    # --- 2. Build metadata message ---
    # Shared memory is raw bytes. The receiver CANNOT infer shape/dtype
    # from the buffer. We must send this metadata out-of-band.
    metadata = {
        "shm_name": shm.name,
        "shape": list(shape),
        "dtype": str(dtype).replace("torch.", ""),  # "float32"
        "byte_size": byte_size,
    }

    # --- 3. Send metadata via ZeroMQ PUSH socket ---
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.bind(endpoint)

    # Signal the receiver that we're ready
    ready_event.set()

    print(f"[Sender PID {pid}] Sending metadata via ZMQ: {json.dumps(metadata)}")
    socket.send_json(metadata)

    # --- 4. Wait for receiver to finish before cleanup ---
    # In production, you'd use a separate ZMQ message or a barrier.
    # Here we just sleep to keep the demo simple.
    time.sleep(2)

    print(f"[Sender PID {pid}] Cleaning up.")
    del tensor
    shm.close()
    shm.unlink()
    socket.close()
    ctx.term()


def receiver_process(endpoint: str, ready_event):
    """
    RECEIVER: Gets metadata from ZeroMQ, then attaches to the
    shared memory block to read/modify the tensor (zero-copy).
    """
    pid = os.getpid()

    # Wait for sender to bind the socket
    ready_event.wait()

    # --- 1. Receive metadata via ZeroMQ PULL socket ---
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PULL)
    socket.connect(endpoint)

    print(f"[Receiver PID {pid}] Waiting for metadata via ZMQ...")
    metadata = socket.recv_json()
    print(f"[Receiver PID {pid}] Received metadata: {json.dumps(metadata)}")

    shm_name = metadata["shm_name"]
    shape = tuple(metadata["shape"])
    dtype = getattr(torch, metadata["dtype"])
    byte_size = metadata["byte_size"]

    # --- 2. Attach to shared memory using the name from ZMQ ---
    existing_shm = shared_memory.SharedMemory(name=shm_name)

    # --- 3. Map to tensor (ZERO-COPY) ---
    tensor = torch.frombuffer(
        existing_shm.buf[:byte_size], dtype=dtype
    ).reshape(shape)

    print(f"[Receiver PID {pid}] Attached. Shape: {tensor.shape}")
    print(f"[Receiver PID {pid}] Before (first 5): {tensor[0, :5]}")

    # --- 4. Modify in-place ---
    tensor.add_(1000)
    print(f"[Receiver PID {pid}] After add_(1000) (first 5): {tensor[0, :5]}")

    # --- 5. Cleanup ---
    del tensor
    existing_shm.close()
    socket.close()
    ctx.term()
    print(f"[Receiver PID {pid}] Done.\n")


def main():
    # ipc:// transport uses Unix domain sockets (local only).
    # For cross-machine, use tcp://192.168.1.1:5555
    endpoint = "ipc:///tmp/zmq_shm_demo"

    ready_event = mp.Event()

    sender = mp.Process(target=sender_process, args=(endpoint, ready_event))
    receiver = mp.Process(target=receiver_process, args=(endpoint, ready_event))

    sender.start()
    receiver.start()

    sender.join()
    receiver.join()

    print("Done. The tensor was modified across PIDs with zero data copy.")


if __name__ == "__main__":
    main()