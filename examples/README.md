# PantheraML Examples

This directory contains comprehensive examples demonstrating various PantheraML use cases and capabilities.

## Available Examples

### 📊 Benchmarking
- **`benchmarking_example.py`** - Complete benchmarking workflow with MMLU, HellaSwag, and ARC
- Shows how to evaluate models and export results

### 🤖 Complete Pipelines
- **`helpsteer2_complete_pipeline.py`** - Full end-to-end pipeline for nvidia/HelpSteer2 dataset
- Covers model loading, training, saving, and inference

## Quick Start

### HelpSteer2 Complete Pipeline

Train a helpful AI assistant on the nvidia/HelpSteer2 dataset:

```bash
# Quick test run (10 steps, 100 samples)
python examples/helpsteer2_complete_pipeline.py --max_steps 10 --max_samples 100

# Full training run
python examples/helpsteer2_complete_pipeline.py --max_steps 1000

# Custom model and settings
python examples/helpsteer2_complete_pipeline.py \
    --model "microsoft/DialoGPT-medium" \
    --max_steps 500 \
    --batch_size 4 \
    --output_dir "./my_model"
```

### Benchmarking Example

Evaluate your trained models:

```bash
# Run MMLU benchmark
python examples/benchmarking_example.py --benchmark mmlu --num_samples 100

# Multiple benchmarks with export
python examples/benchmarking_example.py \
    --benchmark mmlu hellaswag arc \
    --export_format json csv \
    --output_dir ./benchmark_results
```

## Example Features

### HelpSteer2 Pipeline Features
- ✅ Model loading with PantheraML optimizations
- ✅ Dataset formatting and preparation
- ✅ LoRA fine-tuning for efficiency
- ✅ Multiple save formats (LoRA, merged, GGUF)
- ✅ Inference examples and validation
- ✅ Optional benchmarking integration
- ✅ Memory and performance monitoring

### Benchmarking Features
- ✅ Multiple benchmark types (MMLU, HellaSwag, ARC)
- ✅ Flexible model loading
- ✅ Export results in multiple formats
- ✅ Device detection and optimization
- ✅ Progress tracking and logging
- ✅ Error handling and recovery

## Requirements

### Core Dependencies
```bash
pip install torch transformers datasets trl accelerate bitsandbytes
```

### Optional Dependencies
```bash
# For advanced features
pip install flash-attn triton ninja packaging

# For benchmarking
pip install scikit-learn pandas
```

## Usage Patterns

### 1. Development and Testing
```bash
# Quick validation
python examples/helpsteer2_complete_pipeline.py --max_steps 5 --max_samples 50

# Test benchmarking
python examples/benchmarking_example.py --benchmark mmlu --num_samples 10
```

### 2. Production Training
```bash
# Full dataset training
python examples/helpsteer2_complete_pipeline.py \
    --max_steps 1000 \
    --batch_size 8 \
    --run_benchmarks

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/helpsteer2_complete_pipeline.py
```

### 3. Model Evaluation
```bash
# Comprehensive benchmarking
python examples/benchmarking_example.py \
    --model "./my_trained_model" \
    --benchmark mmlu hellaswag arc \
    --num_samples 1000 \
    --export_format json csv html
```

## File Structure

```
examples/
├── README.md                           # This file
├── benchmarking_example.py             # Benchmarking workflow
├── helpsteer2_complete_pipeline.py     # Complete HelpSteer2 pipeline
└── output/                             # Generated outputs
    ├── models/                         # Trained models
    ├── results/                        # Benchmark results
    └── logs/                           # Training logs
```

## Output Formats

### Model Outputs
- **LoRA adapters**: Efficient fine-tuned weights
- **Merged models**: Full model with adapters merged
- **GGUF format**: Optimized for inference
- **Safetensors**: Safe tensor format

### Benchmark Outputs
- **JSON**: Machine-readable results
- **CSV**: Spreadsheet-compatible data
- **HTML**: Human-readable reports
- **Console**: Real-time progress

## Best Practices

### Memory Management
- Start with small batch sizes (1-2)
- Use gradient accumulation for effective larger batches
- Enable 4-bit quantization for memory efficiency
- Monitor GPU memory usage

### Training Efficiency
- Use LoRA for parameter-efficient fine-tuning
- Enable gradient checkpointing for long sequences
- Use mixed precision (FP16/BF16)
- Save checkpoints frequently

### Evaluation
- Always run inference examples after training
- Benchmark on relevant tasks for your use case
- Compare with baseline models
- Validate on held-out data

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size and sequence length
   --batch_size 1 --max_seq_length 1024
   ```

2. **Slow Training**
   ```bash
   # Enable optimizations
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

3. **Import Errors**
   ```bash
   # Check PantheraML installation
   python -c "import pantheraml; print('OK')"
   ```

4. **Poor Model Performance**
   ```bash
   # Increase training data and steps
   --max_steps 1000 --max_samples 10000
   ```

### Validation Scripts

Run the test scripts to validate your setup:

```bash
# Test HelpSteer2 pipeline
python test_helpsteer2_pipeline.py

# Test benchmarking
python validate_benchmarking_api.py
```

## Advanced Customization

### Custom Datasets

Adapt the HelpSteer2 pipeline for your own datasets:

```python
def format_custom_dataset(examples):
    """Adapt this function for your dataset structure"""
    convos = []
    for item in examples:
        # Format your data into chat format
        messages = [
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        convos.append(text)
    return {"text": convos}
```

### Custom Benchmarks

Add your own evaluation tasks:

```python
from pantheraml.benchmarks import BenchmarkRunner

# Create custom benchmark
runner = BenchmarkRunner()
results = runner.run_custom_benchmark(
    model=model,
    tokenizer=tokenizer,
    questions=my_questions,
    answers=my_answers,
    benchmark_name="my_custom_task"
)
```

## Performance Tips

1. **Use the right model size** for your hardware
2. **Enable compilation** for faster inference
3. **Batch inference** when possible
4. **Cache models** to avoid reloading
5. **Profile your code** to identify bottlenecks

## Support

For help with examples:
1. Check the comprehensive guides (HELPSTEER2_PIPELINE_GUIDE.md)
2. Review the API reference (PANTHERAML_COMPLETE_API_REFERENCE.md)
3. Run validation scripts to check your setup
4. Check PantheraML documentation for advanced features
