# Multi-GPU Support for PantheraML

This implementation adds comprehensive multi-GPU support to PantheraML, enabling faster training and inference across multiple GPUs.

## 🚀 Features Added

### Core Multi-GPU Support
- **Distributed Data Parallel (DDP)** training
- **Automatic device mapping** and memory optimization
- **Model parallelism** for large models
- **Mixed precision training** across GPUs
- **Memory monitoring** and management
- **Easy-to-use APIs** with intelligent defaults

### New Files Created

1. **`pantheraml/distributed.py`** - Core distributed training functionality
   - `MultiGPUConfig` class for configuration
   - Setup and cleanup functions for distributed environment
   - Device mapping utilities
   - Communication and synchronization helpers

2. **`examples/multi_gpu_training.py`** - Complete training example
   - Basic and advanced multi-GPU training examples
   - Performance optimization demonstrations
   - Memory management examples

3. **`docs/Multi-GPU-Guide.md`** - Comprehensive documentation
   - Quick start guide
   - Configuration options
   - Performance tips
   - Troubleshooting guide

4. **`tests/test_multi_gpu.py`** - Test suite for multi-GPU functionality
   - GPU detection and setup tests
   - Import verification tests
   - Model loading tests
   - Trainer functionality tests

### Enhanced Files

1. **`pantheraml/trainer.py`** - Enhanced trainer with multi-GPU support
   - New `PantheraMLDistributedTrainer` class
   - Automatic distributed setup
   - Memory monitoring capabilities
   - Gradient synchronization
   - Distributed data loading

2. **`pantheraml/models/loader.py`** - Enhanced model loading
   - Multi-GPU device mapping support
   - Automatic GPU detection
   - Memory optimization for multiple GPUs

3. **`pantheraml/kernels/utils.py`** - Enhanced utilities
   - Multi-GPU stream management
   - GPU memory utilities
   - Device management functions

4. **`pantheraml/__init__.py`** - Updated exports
   - Export new multi-GPU classes and functions
   - Updated documentation strings

5. **`pantheraml/models/_utils.py`** - Removed multi-GPU restrictions
   - Enable Accelerate's multi-GPU support
   - Intelligent distributed detection

## 🎯 Key Components

### PantheraMLDistributedTrainer

The new `PantheraMLDistributedTrainer` extends the standard `PantheraMLTrainer` with:

- Automatic multi-GPU model wrapping
- Distributed data sampling
- Synchronized gradient computation
- Memory monitoring across GPUs
- Intelligent checkpointing

### MultiGPUConfig

Configuration class for customizing multi-GPU behavior:

```python
config = MultiGPUConfig(
    backend="nccl",                    # Communication backend
    auto_device_map=True,              # Automatic device mapping
    use_gradient_checkpointing=True,   # Memory optimization
    find_unused_parameters=False,      # DDP configuration
)
```

### Device Mapping

Intelligent device mapping strategies:
- `"auto"` - Automatic optimization
- `"balanced"` - Balance layers across GPUs
- `"sequential"` - Sequential layer distribution
- Custom mapping with memory limits

## 📊 Performance Benefits

### Expected Improvements

1. **Training Speed**: 2-4x faster with 2-4 GPUs (depends on model size)
2. **Memory Efficiency**: Support for larger models through model parallelism
3. **Scalability**: Easy scaling from 1 to multiple GPUs
4. **Compatibility**: Works with existing Unsloth features (LoRA, 4-bit quantization, etc.)

### Optimization Features

- Automatic mixed precision training
- Gradient accumulation across GPUs
- Memory-efficient model sharding
- Optimized communication patterns
- Intelligent batch size scaling

## 🔧 Usage Examples

### Basic Multi-GPU Training

```python
from pantheraml import FastLanguageModel, PantheraMLDistributedTrainer
from trl import SFTConfig

# Load model with multi-GPU support
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-chat-bnb-4bit",
    use_multi_gpu=True,
    auto_device_map=True,
)

# Create distributed trainer
trainer = PantheraMLDistributedTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(...),
)

# Start training
trainer.train()
```

### Running Multi-GPU Training

```bash
# Automatic (recommended)
python your_script.py

# With torchrun
torchrun --nproc_per_node=4 your_script.py

# With accelerate
accelerate launch --num_processes=4 your_script.py
```

## 🧪 Testing

Run the test suite to verify multi-GPU functionality:

```bash
python tests/test_multi_gpu.py
```

This will test:
- GPU detection and availability
- Import functionality
- Model loading with multi-GPU
- Trainer creation and configuration
- Memory management utilities

## 📋 Requirements

- PyTorch with CUDA support
- Multiple CUDA-capable GPUs
- NCCL for optimal GPU communication
- Existing Unsloth dependencies

## 🔄 Migration Guide

### From Single GPU to Multi-GPU

Minimal changes required for existing code:

```python
# Before (single GPU)
from pantheraml import FastLanguageModel, PantheraMLTrainer

trainer = PantheraMLTrainer(...)

# After (multi-GPU)
from pantheraml import FastLanguageModel, PantheraMLDistributedTrainer

trainer = PantheraMLDistributedTrainer(...)  # Everything else stays the same!
```

### Configuration Migration

```python
# Add multi-GPU configuration
from pantheraml import MultiGPUConfig

config = MultiGPUConfig(
    backend="nccl",
    auto_device_map=True,
)

trainer = PantheraMLDistributedTrainer(
    multi_gpu_config=config,
    # ... existing parameters
)
```

## 🛠️ Implementation Details

### Architecture

1. **Distributed Setup**: Automatic detection and initialization of distributed environment
2. **Model Wrapping**: Intelligent wrapping with DDP or DataParallel based on setup
3. **Data Distribution**: Automatic distributed sampling for training and evaluation
4. **Memory Management**: Cross-GPU memory monitoring and optimization
5. **Communication**: Optimized gradient synchronization and parameter updates

### Compatibility

- ✅ Works with LoRA adapters
- ✅ Compatible with 4-bit and 8-bit quantization
- ✅ Supports gradient checkpointing
- ✅ Works with existing PantheraML optimizations
- ✅ Compatible with vision models
- ✅ Supports mixed precision training

### Error Handling

- Graceful fallback to single GPU if multi-GPU setup fails
- Comprehensive error messages for configuration issues
- Memory overflow detection and suggestions
- Communication timeout handling

## 🚧 Limitations and Future Work

### Current Limitations

1. **Pipeline Parallelism**: Not yet implemented (planned for future release)
2. **Tensor Parallelism**: Basic support, full implementation planned
3. **Dynamic Batching**: Fixed batch sizes across GPUs currently

### Future Enhancements

1. **Advanced Parallelism**: Full pipeline and tensor parallelism
2. **Adaptive Scaling**: Dynamic adjustment of parallelism strategies
3. **Cloud Integration**: Optimizations for cloud multi-GPU setups
4. **Monitoring Dashboard**: Real-time multi-GPU training monitoring

## 📞 Support

For issues with multi-GPU functionality:

1. Check the troubleshooting guide in `docs/Multi-GPU-Guide.md`
2. Run the test suite: `python tests/test_multi_gpu.py`
3. Review the examples in `examples/multi_gpu_training.py`
4. Check environment variables and GPU setup

## 🤝 Contributing

When contributing to multi-GPU functionality:

1. Run the test suite on multi-GPU systems
2. Update documentation for any new features
3. Test with different GPU configurations
4. Verify backward compatibility with single GPU setups

---

This implementation provides a solid foundation for multi-GPU training in PantheraML while maintaining the library's ease of use and performance characteristics.
