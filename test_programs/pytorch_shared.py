import torch
import torch.multiprocessing as mp
import os

def reader_process(tensor):
    pid = os.getpid()
    print(f"[Reader PID {pid}] Current data: {tensor}")
    tensor.add_(100)
    print(f"[Reader PID {pid}] After add_(100): {tensor}")

if __name__ == '__main__':
    pid = os.getpid()

    tensor = torch.arange(6).float().reshape(2, 3)
    tensor.share_memory_()  # Moves to /dev/shm automatically

    print(f"[Creator PID {pid}] Initial: {tensor}")

    p = mp.Process(target=reader_process, args=(tensor,))
    p.start()
    p.join()

    print(f"[Creator PID {pid}] After reader: {tensor}")
    # No manual cleanup needed!