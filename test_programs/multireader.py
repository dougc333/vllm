
#ipc event barrier
#vllm does not do this

import torch
import multiprocessing as mp
import os
import math
import time
from multiprocessing import shared_memory

def worker_inference_loop(worker_id, shm_name, shape, dtype, byte_size, ready_event, done_event):
    """
    Simulates a vLLM GPU Worker Process.
    It loops indefinitely, waiting for the Master to load new inference batches into SHM.
    """
    pid = os.getpid()
    print(f"[Worker {worker_id} (PID {pid})] Initializing and attaching to tensor arena...")
    
    # 1. Map the shared memory segment exactly once during initialization
    shm = mp.shared_memory.SharedMemory(name=shm_name)
    
    try:
        while True:
            # 2. Wait for Master to signal that a new inference batch is ready
            ready_event.wait()
            
            # Check for shutdown signal from Master
            if ready_event.is_set() and done_event.is_set():
                break
                
            # 3. Create a zero-copy PyTorch tensor view over the shared raw memory
            # In a real vLLM setup, this would be wrapped by a CUDA IPC handler if on GPU
            shared_tensor = torch.frombuffer(shm.buf[:byte_size], dtype=dtype).reshape(shape)
            
            # 4. Perform localized inference work (Concurrent Read)
            # We slice or access data without copying it into this process's local heap
            first_elements = shared_tensor[0, :5].tolist()
            print(f"[Worker {worker_id}] Processing Batch. First 5 tokens/metrics: {first_elements}")
            
            # Simulate forward pass computation time
            time.sleep(0.1) 
            
            # 5. Clean up Python references to avoid zombie memory maps
            del shared_tensor
            
            # 6. Signal back to Master that this specific worker is done reading
            done_event.set()
            
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up local process mapping on exit
        shm.close()
        print(f"[Worker {worker_id}] Detached and exited cleanly.")

def main():
    master_pid = os.getpid()
    print(f"[Master Engine {master_pid}] Starting Inference Server...")
    
    # Pipeline Configurations (e.g., Prompt Batch Sizes / Hidden Dimensions)
    num_workers = 3
    shape = (1, 4096)
    dtype = torch.float32
    
    element_size = torch.empty(0, dtype=dtype).element_size()
    byte_size = math.prod(shape) * element_size

    # --- 1. Allocate the Global Shared Memory Arena ---
    # Master creates it. It remains alive until explicitly unlinked by Master at shutdown.
    shm_arena = mp.shared_memory.SharedMemory(create=True, size=byte_size)
    print(f"[Master] Allocated Arena: {shm_arena.name} ({byte_size} bytes)")
    
    # Create the tensor view for the Master to write data into
    master_tensor_view = torch.frombuffer(shm_arena.buf, dtype=dtype).reshape(shape)
    
    # --- 2. Setup Synchronization Synchronization Infrastructure ---
    workers = []
    ready_events = [mp.Event() for _ in range(num_workers)]
    done_events = [mp.Event() for _ in range(num_workers)]
    
    # Spawn the concurrent readers
    for i in range(num_workers):
        p = mp.Process(
            target=worker_inference_loop, 
            args=(i, shm_arena.name, shape, dtype, byte_size, ready_events[i], done_events[i])
        )
        workers.append(p)
        p.start()
        
    # Give workers a brief moment to initialize and attach
    time.sleep(0.5)

    # --- 3. Simulate an Inference Request Stream (e.g., 3 Sequential Batches) ---
    for batch_id in range(3):
        print(f"\n--- [Master] Processing Incoming Prompt Request Batch #{batch_id} ---")
        
        # Reset synchronization flags
        for e in done_events: e.clear()
        for e in ready_events: e.clear()
            
        # Master writes data directly to the Shared Memory Arena (In-place copy)
        # In vLLM, this represents Kv-Cache updates or input token embeddings
        master_tensor_view[:] = torch.randn(shape) * (batch_id + 1)
        
        print(f"[Master] Batch #{batch_id} written to SHM. Broadcasting to workers...")
        
        # Notify all concurrent readers that data is stabilized and ready
        for e in ready_events: e.set()
            
        # Wait for all concurrent readers to finish their forward passes
        for e in done_events: e.wait()
        print(f"[Master] All {num_workers} workers processed Batch #{batch_id} successfully.")

    # --- 4. Graceful Engine Teardown Sequence ---
    print("\n--- [Master] Initiating Engine Shutdown ---")
    for i in range(num_workers):
        ready_events[i].set()
        done_events[i].set() # Poison pill combination to break loops
        
    for p in workers:
        p.join()
        
    # Master safely unlinks memory now that all workers have detached
    del master_tensor_view
    shm_arena.close()
    shm_arena.unlink()
    print("[Master] Shared Memory Arena unlinked. System shutdown complete.")

if __name__ == '__main__':
    main()