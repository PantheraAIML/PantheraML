# Multi-GPU Support in PantheraML

*Built on the excellent foundation of Unsloth*

## Credits & Acknowledgments

PantheraML's multi-GPU capabilities are built upon the outstanding work of the [Unsloth team](https://github.com/unslothai/unsloth). We extend our sincere gratitude to Daniel Han-Chen and the entire Unsloth community for creating such an efficient foundation for LLM fine-tuning.

**Original Unsloth:** https://github.com/unslothai/unsloth

---

PantheraML now includes comprehensive multi-GPU support for faster training and inference! This guide covers how to use the new multi-GPU features.

## Overview

The multi-GPU support includes:

- **Distributed Data Parallel (DDP)** training across multiple GPUs
- **Automatic device mapping** for optimal memory usage
- **Model parallelism** for large models that don't fit on a single GPU
- **Mixed precision training** with automatic optimization
- **Memory management** across multiple GPUs
- **Easy-to-use APIs** with sensible defaults

## Quick Start

### Basic Multi-GPU Training

```python
from pantheraml import FastLanguageModel, PantheraMLDistributedTrainer, MultiGPUConfig
from trl import SFTConfig
from datasets import load_dataset

# Load model with multi-GPU support
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-chat-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    use_multi_gpu=True,  # Enable multi-GPU
    auto_device_map=True,  # Automatic device mapping
)

# Prepare model for training
FastLanguageModel.for_training(model)

# Load dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Create distributed trainer
trainer = PantheraMLDistributedTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="./outputs",
    ),
)

# Start training
trainer.train()
```

## Configuration Options

### MultiGPUConfig

Configure multi-GPU behavior with `MultiGPUConfig`:

```python
from pantheraml import MultiGPUConfig

config = MultiGPUConfig(
    backend="nccl",  # Communication backend (nccl for NVIDIA, gloo for CPU)
    timeout_minutes=30,  # Timeout for distributed operations
    auto_device_map=True,  # Automatic device mapping
    tensor_parallel=False,  # Enable tensor parallelism for very large models
    pipeline_parallel=False,  # Enable pipeline parallelism
    use_gradient_checkpointing=True,  # Save memory with gradient checkpointing
    find_unused_parameters=False,  # Find unused parameters (set True if needed)
)

trainer = PantheraMLDistributedTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    multi_gpu_config=config,
)
```

### Device Mapping Strategies

Control how the model is distributed across GPUs:

```python
# Automatic device mapping (recommended)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-chat-bnb-4bit",
    device_map="auto",  # or "balanced", "sequential"
    use_multi_gpu=True,
)

# Manual memory limits per GPU
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-chat-bnb-4bit",
    device_map="balanced",
    max_memory={0: "10GB", 1: "10GB", 2: "8GB", 3: "8GB"},
    use_multi_gpu=True,
)
```

## Running Multi-GPU Training

### Single Node, Multiple GPUs

#### Method 1: Automatic Setup (Recommended)
```bash
python your_training_script.py
```
PantheraML automatically detects and uses all available GPUs.

#### Method 2: Using torchrun
```bash
torchrun --nproc_per_node=4 your_training_script.py
```

#### Method 3: Using accelerate
```bash
accelerate launch --num_processes=4 your_training_script.py
```

### Multiple Nodes

#### Using torchrun
```bash
# On node 0 (master)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.100" --master_port=12355 your_training_script.py

# On node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="192.168.1.100" --master_port=12355 your_training_script.py
```

#### Using SLURM
```bash
srun --nodes=2 --gpus-per-node=4 python your_training_script.py
```

## Advanced Features

### Memory Monitoring

Monitor GPU memory usage across all devices:

```python
trainer = UnslothDistributedTrainer(...)

# Print memory stats
trainer.print_memory_stats()

# Get memory usage programmatically
memory_stats = trainer.get_model_memory_usage()
print(memory_stats)
```

### Custom Distributed Setup

For advanced use cases, manually setup distributed environment:

```python
from pantheraml import setup_multi_gpu, cleanup_multi_gpu, distributed_context

# Manual setup
config = setup_multi_gpu(MultiGPUConfig())

# Your training code here
# ...

# Cleanup
cleanup_multi_gpu()

# Or use context manager
with distributed_context(MultiGPUConfig()) as config:
    # Your training code here
    pass
```

### Mixed Precision Training

Unsloth automatically optimizes mixed precision across GPUs:

```python
training_args = SFTConfig(
    fp16=not torch.cuda.is_bf16_supported(),  # Use FP16 if BF16 not available
    bf16=torch.cuda.is_bf16_supported(),      # Use BF16 if available
    # ... other args
)
```

## Utility Functions

### Check Multi-GPU Availability

```python
from pantheraml import is_distributed_available, get_world_size, get_rank, is_main_process

print(f"Distributed available: {is_distributed_available()}")
print(f"World size: {get_world_size()}")
print(f"Current rank: {get_rank()}")
print(f"Is main process: {is_main_process()}")
```

### Device Management

```python
from pantheraml.kernels.utils import (
    get_current_gpu_index,
    set_gpu_device,
    get_gpu_memory_info,
    clear_gpu_cache
)

# Get current GPU
current_gpu = get_current_gpu_index()

# Set GPU device
set_gpu_device(1)

# Get memory info
memory_info = get_gpu_memory_info(0)
print(f"GPU 0 memory: {memory_info}")

# Clear cache
clear_gpu_cache()
```

## Performance Tips

### 1. Batch Size Optimization
- Start with `per_device_train_batch_size=1` and increase gradually
- Use `gradient_accumulation_steps` to maintain effective batch size
- Total effective batch size = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`

### 2. Memory Management
- Use `load_in_4bit=True` to reduce memory usage
- Enable `gradient_checkpointing=True` to save memory
- Set appropriate `max_memory` limits per GPU

### 3. Communication Backend
- Use `nccl` backend for NVIDIA GPUs (fastest)
- Use `gloo` backend for CPU-only or mixed setups
- Use `mpi` backend for high-performance computing environments

### 4. Model Sharding
- For very large models, use `device_map="sequential"`
- For balanced workloads, use `device_map="balanced"`
- For automatic optimization, use `device_map="auto"`

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```python
# Solutions:
# - Reduce batch size
# - Increase gradient accumulation steps
# - Enable gradient checkpointing
# - Use smaller model or quantization

trainer = UnslothDistributedTrainer(
    args=SFTConfig(
        per_device_train_batch_size=1,  # Reduce batch size
        gradient_accumulation_steps=8,  # Increase accumulation
        gradient_checkpointing=True,    # Save memory
    ),
    multi_gpu_config=MultiGPUConfig(
        use_gradient_checkpointing=True,
    ),
)
```

#### 2. Slow Training
```python
# Solutions:
# - Increase batch size if memory allows
# - Optimize device mapping
# - Use faster data loading

training_args = SFTConfig(
    per_device_train_batch_size=4,  # Increase if memory allows
    dataloader_num_workers=4,       # Parallel data loading
    dataloader_pin_memory=True,     # Faster GPU transfer
)
```

#### 3. Uneven GPU Utilization
```python
# Use balanced device mapping
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-chat-bnb-4bit",
    device_map="balanced",  # Balance across GPUs
    use_multi_gpu=True,
)
```

### Environment Variables

Useful environment variables for debugging:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify which GPUs to use
export TORCH_DISTRIBUTED_DEBUG=INFO  # Enable distributed debugging
export NCCL_DEBUG=INFO               # Enable NCCL debugging
export PYTHONPATH=/path/to/unsloth:$PYTHONPATH
```

## Examples

See the `examples/multi_gpu_training.py` file for complete working examples:

```bash
# Basic multi-GPU training
python examples/multi_gpu_training.py

# Advanced configuration
python examples/multi_gpu_training.py --advanced
```

## API Reference

### Classes

- `UnslothDistributedTrainer`: Enhanced trainer with multi-GPU support
- `MultiGPUConfig`: Configuration for multi-GPU training

### Functions

- `setup_multi_gpu()`: Setup distributed environment
- `cleanup_multi_gpu()`: Cleanup distributed environment
- `is_distributed_available()`: Check if distributed training is available
- `get_world_size()`: Get number of processes
- `get_rank()`: Get current process rank
- `is_main_process()`: Check if this is the main process

For more detailed API documentation, see the docstrings in the source code.
