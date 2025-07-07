# 🧪 Experimental TPU Support in PantheraML

*Built on the excellent foundation of Unsloth*

## ⚠️ EXPERIMENTAL WARNING

**TPU support in PantheraML is experimental and comes with limitations:**

- May not work with all model architectures
- Limited debugging and error reporting
- Performance may vary significantly
- Some features may not be supported
- Requires additional dependencies (torch_xla)

**Use at your own risk and expect potential issues.**

## Credits & Acknowledgments

PantheraML's TPU capabilities are built upon the outstanding work of the [Unsloth team](https://github.com/unslothai/unsloth). The experimental TPU support extends Unsloth's foundation to work with Google's Tensor Processing Units.

**Original Unsloth:** https://github.com/unslothai/unsloth

---

## Prerequisites

### 1. Install torch_xla
```bash
pip install torch_xla
```

### 2. TPU Access
You need access to TPU resources:
- **Google Colab**: Select TPU runtime in Runtime > Change runtime type
- **Google Cloud Platform**: Use TPU VMs or TPU Pods
- **Kaggle**: Select TPU accelerator in notebook settings

### 3. Environment Setup
```python
# Check TPU availability
from pantheraml import is_tpu_available
print(f"TPU Available: {is_tpu_available()}")
```

## Basic Usage

### 1. Setup TPU Configuration

```python
from pantheraml import setup_multi_tpu, MultiTPUConfig

# Setup experimental TPU environment
config = setup_multi_tpu(MultiTPUConfig(
    num_cores=8,
    auto_device_map=True,
    use_gradient_checkpointing=True,
))
```

### 2. Load Model with TPU Support

```python
from pantheraml import FastLanguageModel

# Load model with experimental TPU support
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama-chat-bnb-4bit",
    max_seq_length=512,
    dtype=None,  # Let TPU decide
    load_in_4bit=False,  # Disable quantization for TPU
    use_tpu=True,  # Enable experimental TPU support
    tpu_cores=8,
)
```

### 3. Create TPU Trainer

```python
from pantheraml import PantheraMLTPUTrainer

# Create experimental TPU trainer
trainer = PantheraMLTPUTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    tpu_config=config,
)

# Train the model
trainer.train()
```

## Complete Example

```python
#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

# Import experimental TPU support
from pantheraml import (
    FastLanguageModel,
    PantheraMLTPUTrainer, 
    setup_multi_tpu,
    MultiTPUConfig,
    is_tpu_available,
)

def main():
    print("🧪 EXPERIMENTAL: PantheraML TPU Training")
    
    if not is_tpu_available():
        print("❌ TPU not available")
        return
    
    # Setup TPU
    config = setup_multi_tpu()
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/tinyllama-chat-bnb-4bit",
        max_seq_length=512,
        use_tpu=True,
        tpu_cores=8,
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
    )
    
    # Create trainer
    trainer = PantheraMLTPUTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,  # Your dataset
        tpu_config=config,
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
```

## TPU-Specific Configuration

### MultiTPUConfig Options

```python
config = MultiTPUConfig(
    num_cores=8,                      # Number of TPU cores
    auto_device_map=True,             # Automatic device placement
    use_gradient_checkpointing=True,  # Memory optimization
)
```

### Environment Variables

```python
import os

# TPU-specific optimizations
os.environ["XLA_USE_BF16"] = "1"                    # Enable bfloat16
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"  # Limit allocator
```

## Limitations and Known Issues

### Current Limitations

1. **Model Support**: Not all models are compatible with TPUs
2. **Quantization**: 4-bit/8-bit quantization not supported on TPUs
3. **Memory**: TPU memory management differs from GPU
4. **Debugging**: Limited error reporting and debugging tools
5. **Performance**: May be slower than GPU for some workloads

### Recommended Settings

- Use smaller models (< 7B parameters)
- Reduce sequence length (≤ 512 tokens)
- Disable quantization (`load_in_4bit=False`)
- Use smaller LoRA ranks (r=8-16)
- Enable gradient checkpointing

### Troubleshooting

**Common Issues:**

1. **ImportError: torch_xla not found**
   ```bash
   pip install torch_xla
   ```

2. **TPU not detected**
   - Ensure TPU runtime is selected (Colab/Kaggle)
   - Check TPU availability with `is_tpu_available()`

3. **Out of memory errors**
   - Reduce batch size
   - Use gradient checkpointing
   - Reduce sequence length

4. **Slow training**
   - TPU performance varies by workload
   - Consider GPU for comparison

## Utility Functions

### TPU Status and Monitoring

```python
from pantheraml import (
    get_tpu_rank,
    get_tpu_world_size, 
    is_tpu_main_process,
    synchronize_tpu,
)

# Check TPU status
print(f"TPU Rank: {get_tpu_rank()}")
print(f"TPU World Size: {get_tpu_world_size()}")
print(f"Is Main Process: {is_tpu_main_process()}")

# Synchronize TPU operations
synchronize_tpu()
```

### Memory Monitoring

```python
# Print TPU memory stats
trainer.print_memory_stats()
```

## Performance Considerations

### When to Use TPUs

**Good for:**
- Large batch training
- Models that fit well in TPU memory
- Workloads that can utilize all TPU cores
- Research and experimentation

**Consider GPU instead for:**
- Interactive development
- Small batch inference
- Models requiring frequent debugging
- Production deployments (for now)

### Optimization Tips

1. **Batch Size**: Use larger batch sizes to maximize TPU utilization
2. **Sequence Length**: Keep sequences reasonably short (≤ 1024)
3. **Model Size**: Start with smaller models to verify functionality
4. **Gradient Accumulation**: Use gradient accumulation for effective large batches

## Examples and Resources

### Example Scripts
- `examples/experimental_tpu_training.py` - Basic TPU training example
- `examples/tpu_benchmark.py` - TPU vs GPU performance comparison (if available)

### External Resources
- [Google TPU Documentation](https://cloud.google.com/tpu/docs)
- [PyTorch XLA Documentation](https://pytorch.org/xla/)
- [Colab TPU Tutorial](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb)

## Feedback and Issues

Since TPU support is experimental, we welcome feedback:

1. **Performance issues**: Compare with GPU baselines
2. **Compatibility problems**: Report unsupported models/features  
3. **Errors and crashes**: Provide full error traces
4. **Feature requests**: Suggest improvements

**Note**: Remember to acknowledge that this builds upon Unsloth's excellent foundation when reporting issues or discussing the experimental TPU features.

## Roadmap

Future improvements may include:
- Better error handling and debugging
- Support for more model architectures
- Performance optimizations
- Integration with distributed training
- Production-ready features

**Disclaimer**: This is experimental software. Use in production environments is not recommended without thorough testing.
