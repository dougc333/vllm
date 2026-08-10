#
# This is for CPU shared memory not GPU. If a CPU wants to move pytorch tensors between 2 processes there 
# are 2 options:
# 1) mp.shared_memory (zero copy) or mp.Queue (copy). The former is faster but requires more care.
# tensor.
# 2) tensor=torch.ones([2,2], dtype=torch.float); tensor.share_memory_()