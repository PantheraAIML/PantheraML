# Kaggle Usage Guide for PantheraML HelpSteer2 Pipeline

This guide shows how to use the HelpSteer2 complete pipeline in Kaggle notebooks without command-line interface.

## Quick Start in Kaggle

### 1. Import and Run Quick Test

```python
# Import the pipeline functions
from helpsteer2_complete_pipeline import kaggle_quick_test, kaggle_full_training, kaggle_inference_only, run_kaggle_pipeline

# Quick test (recommended first run)
results = kaggle_quick_test()
print("Quick test completed!")
print(f"Training completed: {results['training_completed']}")
print(f"Model saved to: {results['model_path']}")
```

### 2. Full Training Run

```python
# Full training with more data and steps
results = kaggle_full_training()
print("Full training completed!")
```

### 3. Custom Configuration

```python
# Custom training configuration
results = run_kaggle_pipeline(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=1024,  # Smaller for memory efficiency
    batch_size=1,         # Conservative for Kaggle GPU limits
    max_steps=200,
    max_samples=1000,
    multi_gpu=True,       # Use both GPUs if available
    learning_rate=1e-4,
    output_dir="./my_qwen_model"
)
```

### 4. Inference Only

```python
# If you already have a trained model
results = kaggle_inference_only()
print("Inference completed!")
print(f"Results: {results['inference_results']}")
```

## Kaggle-Specific Optimizations

### Memory Management
```python
# For Kaggle's GPU memory limits
results = run_kaggle_pipeline(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    batch_size=1,           # Start small
    max_seq_length=1024,    # Reduce if needed
    gradient_accumulation_steps=8,  # Maintain effective batch size
    max_samples=2000        # Limit dataset size
)
```

### Multi-GPU on Kaggle
```python
# Kaggle provides 2x T4 GPUs
results = run_kaggle_pipeline(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    multi_gpu=True,         # Use both GPUs
    batch_size=2,           # Per GPU
    max_steps=300
)
```

### Production Training
```python
# For serious training runs
results = run_kaggle_pipeline(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_steps=1000,
    max_samples=10000,      # Use more data
    batch_size=2,
    multi_gpu=True,
    run_benchmarks=True,    # Evaluate performance
    learning_rate=2e-4,
    warmup_steps=50
)
```

## Function Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"Qwen/Qwen2.5-0.5B-Instruct"` | HuggingFace model to fine-tune |
| `max_seq_length` | `2048` | Maximum sequence length |
| `batch_size` | `2` | Training batch size per GPU |
| `max_steps` | `100` | Maximum training steps |
| `max_samples` | `None` | Limit dataset samples (None = all) |
| `output_dir` | `"./pantheraml_helpsteer2_model"` | Output directory |
| `skip_training` | `False` | Skip training, inference only |
| `run_benchmarks` | `False` | Run model evaluation |
| `multi_gpu` | `True` | Use multiple GPUs if available |
| `learning_rate` | `2e-4` | Learning rate |
| `gradient_accumulation_steps` | `4` | Gradient accumulation |
| `warmup_steps` | `5` | Learning rate warmup |

## Return Values

The functions return a dictionary with:
```python
{
    "model_path": "./pantheraml_helpsteer2_model",
    "merged_model_path": "./pantheraml_helpsteer2_model_merged", 
    "gguf_model_path": "./pantheraml_helpsteer2_model_gguf",
    "inference_results": [...],  # List of inference examples
    "benchmark_results": {...}, # Benchmark scores (if run)
    "training_completed": True   # Training success status
}
```

## Error Handling

```python
try:
    results = run_kaggle_pipeline(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_steps=100
    )
    
    if results["training_completed"]:
        print("✅ Training successful!")
    else:
        print("⚠️ Training had issues, check results")
        
    if "training_error" in results:
        print(f"Training error: {results['training_error']}")
        
    if "inference_error" in results:
        print(f"Inference error: {results['inference_error']}")
        
except Exception as e:
    print(f"Pipeline error: {e}")
```

## Memory Optimization Tips

1. **Start Small**: Use `batch_size=1` and `max_samples=100` first
2. **Reduce Sequence Length**: Try `max_seq_length=1024` or `512`
3. **Use Gradient Accumulation**: Increase `gradient_accumulation_steps`
4. **Monitor Memory**: Check GPU usage in Kaggle
5. **Clear Cache**: Use `torch.cuda.empty_cache()` between runs

## Example Kaggle Notebook Cell

```python
# Cell 1: Install dependencies (if needed)
# !pip install transformers datasets trl accelerate bitsandbytes

# Cell 2: Import and run
from helpsteer2_complete_pipeline import kaggle_quick_test

# Quick test run
print("Starting PantheraML HelpSteer2 pipeline...")
results = kaggle_quick_test()

print(f"Training completed: {results['training_completed']}")
print(f"Model path: {results['model_path']}")

if results['inference_results']:
    print(f"Generated {len(results['inference_results'])} inference examples")
    for i, result in enumerate(results['inference_results'][:2]):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {result['prompt']}")
        print(f"Response: {result['response'][:100]}...")
```

## Advanced Usage

### Custom Dataset Processing
```python
# Modify the dataset processing before training
def custom_kaggle_run():
    # You can modify the pipeline functions as needed
    results = run_kaggle_pipeline(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_samples=500,  # Small for testing
        batch_size=1
    )
    return results

results = custom_kaggle_run()
```

### Monitoring Training
```python
import torch

# Monitor GPU memory during training
def monitor_training():
    results = run_kaggle_pipeline(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_steps=50,
        batch_size=1
    )
    
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    return results

results = monitor_training()
```
