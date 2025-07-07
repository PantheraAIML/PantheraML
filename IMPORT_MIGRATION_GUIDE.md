# PantheraML Import Migration Guide

## Package Name Change: `unsloth` → `pantheraml`

PantheraML is now imported as `pantheraml` instead of `unsloth`. All functionality is preserved and enhanced with new multi-GPU and TPU features.

## Migration Examples

### Basic Imports

**Before (Unsloth):**
```python
from unsloth import FastLanguageModel
from unsloth import FastMambaModel
```

**After (PantheraML):**
```python
from pantheraml import FastLanguageModel
from pantheraml import FastMambaModel
```

### Trainer Classes

**Before (Unsloth):**
```python
from unsloth import FastLanguageModel
trainer = SFTTrainer(...)  # from trl
```

**After (PantheraML):**
```python
from pantheraml import FastLanguageModel, PantheraMLTrainer
trainer = PantheraMLTrainer(...)  # Enhanced with multi-GPU support
```

### Multi-GPU Training (NEW!)

**PantheraML Only:**
```python
from pantheraml import (
    FastLanguageModel,
    PantheraMLDistributedTrainer,
    MultiGPUConfig,
    setup_multi_gpu,
    cleanup_multi_gpu
)

# Setup multi-GPU
gpu_config = MultiGPUConfig(num_gpus=4, auto_device_map=True)
setup_multi_gpu(gpu_config)

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="microsoft/DialoGPT-medium",
        use_multi_gpu=True,
        auto_device_map=True,
    )
    
    trainer = PantheraMLDistributedTrainer(model=model, ...)
    trainer.train()
finally:
    cleanup_multi_gpu()
```

### Experimental TPU Training (NEW!)

**PantheraML Only:**
```python
from pantheraml import (
    FastLanguageModel,
    PantheraMLTPUTrainer,
    MultiTPUConfig,
    setup_multi_tpu,
    is_tpu_available
)

if is_tpu_available():
    tpu_config = MultiTPUConfig(num_cores=8)
    setup_multi_tpu(tpu_config)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="microsoft/DialoGPT-medium",
        use_tpu=True,
    )
    
    trainer = PantheraMLTPUTrainer(model=model, ...)
    trainer.train()
```

### Utility Imports

**Before (Unsloth):**
```python
from unsloth import is_bfloat16_supported
from unsloth.kernels.utils import get_gpu_memory_info
```

**After (PantheraML):**
```python
from pantheraml import is_bfloat16_supported
from pantheraml.kernels.utils import get_device_memory_info  # Now supports GPU/TPU/CPU
```

### CLI Usage

**Before (Unsloth):**
```bash
python unsloth-cli.py --help
```

**After (PantheraML):**
```bash
python pantheraml-cli.py --help
```

### Installation

**Before (Unsloth):**
```bash
pip install unsloth
```

**After (PantheraML):**
```bash
pip install pantheraml
```

## What's Preserved?

✅ **All original Unsloth functionality**
✅ **Same model support** (Llama, Mistral, Qwen, etc.)
✅ **Same performance optimizations**
✅ **Same memory efficiency**
✅ **Compatible training arguments**

## What's New?

🚀 **Multi-GPU distributed training**
🧪 **Experimental TPU support**
🔧 **Enhanced memory management**
💪 **Extended device compatibility**
📊 **Better monitoring and logging**

## Migration Steps

1. **Uninstall Unsloth:**
   ```bash
   pip uninstall unsloth
   ```

2. **Install PantheraML:**
   ```bash
   pip install pantheraml
   ```

3. **Update imports in your code:**
   - Replace `from unsloth` with `from pantheraml`
   - Replace `import unsloth` with `import pantheraml`

4. **Optional: Use new features:**
   - Add multi-GPU support with `MultiGPUConfig`
   - Try experimental TPU support with `MultiTPUConfig`
   - Use enhanced trainers: `PantheraMLDistributedTrainer`, `PantheraMLTPUTrainer`

## Example: Complete Migration

**Before (Unsloth script):**
```python
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/DialoGPT-medium",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(...)
)

trainer.train()
```

**After (PantheraML script with multi-GPU):**
```python
from pantheraml import FastLanguageModel, PantheraMLDistributedTrainer, MultiGPUConfig
import torch

# Setup multi-GPU if available
if torch.cuda.device_count() > 1:
    from pantheraml import setup_multi_gpu, cleanup_multi_gpu
    gpu_config = MultiGPUConfig(num_gpus=torch.cuda.device_count(), auto_device_map=True)
    setup_multi_gpu(gpu_config)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/DialoGPT-medium",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    use_multi_gpu=torch.cuda.device_count() > 1,
)

trainer = PantheraMLDistributedTrainer(  # Enhanced trainer with multi-GPU support
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(...)
)

trainer.train()

# Cleanup if using multi-GPU
if torch.cuda.device_count() > 1:
    cleanup_multi_gpu()
```

## Credits

PantheraML extends the excellent work of the Unsloth team:
- **Original Unsloth:** https://github.com/unslothai/unsloth
- **PantheraML:** Adds multi-GPU and TPU support while preserving all original functionality

Happy fine-tuning with PantheraML! 🚀
