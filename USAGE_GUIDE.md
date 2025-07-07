"""
PantheraML Quick Start Guide
===========================

This document shows how to use PantheraML with the new package structure.

## Installation

```bash
pip install pantheraml
```

## Basic Usage

### Single GPU Training (replaces original Unsloth functionality)
```python
import torch
from pantheraml import FastLanguageModel

# Load model with PantheraML optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/DialoGPT-medium",
    max_seq_length=2048,
    dtype=None,  # Auto-detect best dtype
    load_in_4bit=True,
)

# Your training code here...
```

### Multi-GPU Training (NEW!)
```python
import torch
from pantheraml import (
    FastLanguageModel, 
    PantheraMLDistributedTrainer,
    MultiGPUConfig,
    setup_multi_gpu,
    cleanup_multi_gpu
)

# Setup multi-GPU configuration
gpu_config = MultiGPUConfig(
    num_gpus=4,
    auto_device_map=True,
    use_deepspeed=False
)

# Initialize distributed training
setup_multi_gpu(gpu_config)

try:
    # Load model across multiple GPUs
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="microsoft/DialoGPT-medium",
        max_seq_length=2048,
        use_multi_gpu=True,
        auto_device_map=True,
    )
    
    # Use distributed trainer
    trainer = PantheraMLDistributedTrainer(
        model=model,
        # ... other training arguments
    )
    
    trainer.train()
    
finally:
    cleanup_multi_gpu()
```

### Experimental TPU Training (NEW!)
```python
import torch
from pantheraml import (
    FastLanguageModel,
    PantheraMLTPUTrainer, 
    MultiTPUConfig,
    setup_multi_tpu,
    is_tpu_available
)

if is_tpu_available():
    print("🧪 EXPERIMENTAL: TPU detected")
    
    # Setup TPU configuration
    tpu_config = MultiTPUConfig(
        num_cores=8,
        mesh_shape=(2, 4)
    )
    
    setup_multi_tpu(tpu_config)
    
    # Load model for TPU
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="microsoft/DialoGPT-medium",
        use_tpu=True,
    )
    
    # Use TPU trainer
    trainer = PantheraMLTPUTrainer(
        model=model,
        # ... training arguments
    )
    
    trainer.train()
```

## CLI Usage

```bash
# Basic training
pantheraml-cli.py --model_name="microsoft/DialoGPT-medium" --dataset="your_dataset"

# Multi-GPU training (automatic detection)
torchrun --nproc_per_node=4 pantheraml-cli.py --model_name="microsoft/DialoGPT-medium" --dataset="your_dataset"

# Show all options
pantheraml-cli.py --help
```

## Migration from Unsloth

If you're migrating from the original Unsloth, simply change your imports:

### Before (Unsloth)
```python
from unsloth import FastLanguageModel
from unsloth import SFTTrainer as UnslothSFTTrainer
```

### After (PantheraML)
```python
from pantheraml import FastLanguageModel
from pantheraml import PantheraMLTrainer  # Enhanced with multi-GPU support
```

## What's New in PantheraML?

1. **Multi-GPU Support**: Distributed training across multiple GPUs
2. **TPU Support**: Experimental support for Google Cloud TPUs  
3. **Built-in Benchmarking**: MMLU, HellaSwag, ARC with auto-export
4. **Enhanced Memory Management**: Better memory optimization across devices
5. **Improved CLI**: More features and better UX
6. **Better Documentation**: Comprehensive guides and examples

## 🧪 Built-in Benchmarking (NEW!)

```python
from pantheraml import benchmark_mmlu, PantheraBench

# Quick MMLU benchmark (your requested syntax!)
model, tokenizer = FastLanguageModel.from_pretrained(...)
result = benchmark_mmlu(model, tokenizer, export=True)

# Comprehensive benchmark suite
bench = PantheraBench(model, tokenizer)
all_results = bench.run_suite(export=True)
```

**Supported Benchmarks:**
- MMLU (57 academic subjects)
- HellaSwag (commonsense reasoning)  
- ARC (AI2 reasoning challenge)
- Automatic JSON/CSV export
- Device performance tracking

## Credits

PantheraML extends the excellent work of the Unsloth team:
- Original Unsloth: https://github.com/unslothai/unsloth
- All original Unsloth functionality is preserved and enhanced

Happy fine-tuning with PantheraML! 🚀
"""
