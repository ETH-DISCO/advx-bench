import torch

assert torch.cuda.is_available(), "CUDA is not available"

gpu_count = torch.cuda.device_count()
print(f"Number of available GPUs: {gpu_count}")

for i in range(gpu_count):
    torch.cuda.set_device(i)
    
    props = torch.cuda.get_device_properties(i)
    
    print(f"\nGPU {i}:")
    print(f"  Name: {props.name}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"  Multi-Processor Count: {props.multi_processor_count}")
    
    memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
    memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
    print(f"  Memory Allocated: {memory_allocated:.2f} MB")
    print(f"  Memory Reserved: {memory_reserved:.2f} MB")
    
    utilization = torch.cuda.utilization(i)
    print(f"  GPU Utilization: {utilization}%")
