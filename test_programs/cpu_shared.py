# /dev/shm is a temp file system for posix shared memory. Memory with file system ops. 
#zero copy ipc process A writes to the fs and process B has immediate visibility. 


# this wont work for multiple concurrent readers

import torch
import multiprocessing as mp
import os
import math
from multiprocessing import shared_memory

def reader_process(shm_name, shape, dtype, byte_size):
    """This runs in a completely separate PID."""
    pid = os.getpid()
    print(f"[Reader PID {pid}] Attaching to shared memory: {shm_name}")
    
    # 1. Attach to the existing shared memory block by its name
    existing_shm = mp.shared_memory.SharedMemory(name=shm_name)
    
    # 2. Map the raw bytes to a PyTorch tensor (ZERO-COPY)
    # torch.frombuffer reads the memoryview without copying the underlying data
    shared_tensor = torch.frombuffer(existing_shm.buf[:byte_size], dtype=dtype).reshape(shape)
    print(f"[Reader PID {pid}] Current data:\n{shared_tensor.shape}")
    print(f"[Reader PID {pid}] Current data:\n{shared_tensor[:10]}")
        
    
    # 3. Modify the tensor IN-PLACE
    #print(f"[Reader PID {pid}] Adding 100 to all elements...")
    #shared_tensor.add_(100)  # The _ means in-place operation
    # how does reader know shape if this is shared memory? no way to pass tensor type tnrougb
    print(f"[Reader PID {pid}] Modified data:\n{shared_tensor.shape}")

    # 4. Detach from the memory (DO NOT unlink, the creator owns the lifecycle)
    existing_shm.close()
    print(f"[Reader PID {pid}] Done and detached.\n")

def main():

    pid = os.getpid()
    
    # --- 1. Define Tensor Properties ---
    shape = (1, 4096)
    dtype = torch.float32
    
    # Calculate exact byte size needed (float32 = 4 bytes. 2*3 = 6 elements. 6*4 = 24 bytes)
    element_size = torch.empty(0, dtype=dtype).element_size()
    num_elements = math.prod(shape)
    byte_size = num_elements * element_size

    # --- 2. Create the Shared Memory Block ---
    # This physically creates a file in /dev/shm
    shm = mp.shared_memory.SharedMemory(create=True, size=byte_size)

    print(f"[Creator PID {pid}] Created shared memory block named: '{shm.name}'")
    print(f"[Creator PID {pid}] Requested: {byte_size} bytes OS Allocated:{shm.size} bytes\n")
    
    # --- 3. Create a PyTorch Tensor backed by this shared memory ---
    tensor_a = torch.frombuffer(shm.buf, dtype=dtype).reshape(shape)
    
    # Initialize the data
    tensor_a[:] = torch.arange(num_elements, dtype=dtype).reshape(shape)
    print(f"[Creator PID {pid}] Initialized tensor a shape:\n{tensor_a.shape}\n")
    
    # --- 4. Spawn the Second Process ---
    # We MUST pass the name, shape, and dtype. Shared memory is just raw bytes;
    # PyTorch doesn't know what the shape/dtype is unless we tell it.
    p = mp.Process(target=reader_process, args=(shm.name, shape, dtype, byte_size))
    p.start()
    
    # Wait for the reader to finish its work
    p.join()
    
    # --- 5. Verify the Zero-Copy Magic ---
    print(f"[Creator PID {pid}] Reading tensor after Reader modified it:")
    print(f"{tensor_a}")
    print("\nNotice the values changed instantly with ZERO network/socket transfer!")
    
    # --- 6. Cleanup (CRITICAL) ---
    shm.close()  # Close the local mapping
    shm.unlink() # DELETE the file in /dev/shm so it doesn't leak RAM
    print(f"\n[Creator PID {pid}] Cleaned up /dev/shm.")

if __name__ == '__main__':
    main()