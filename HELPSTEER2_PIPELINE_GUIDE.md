# PantheraML HelpSteer2 Complete Pipeline Guide

This guide covers the complete pipeline for using PantheraML to fine-tune and deploy models on the nvidia/HelpSteer2 dataset.

## Overview

The `helpsteer2_complete_pipeline.py` script provides a full end-to-end workflow:

1. **Model Loading**: Load pre-trained models with PantheraML optimizations
2. **Dataset Preparation**: Format nvidia/HelpSteer2 for training
3. **Fine-tuning**: Train with LoRA adapters for efficiency
4. **Model Saving**: Save in multiple formats (LoRA, merged, GGUF)
5. **Inference**: Load and run inference with examples
6. **Benchmarking**: Optional model evaluation

## Quick Start

### Basic Usage

```bash
# Run with default settings (small scale for testing)
python examples/helpsteer2_complete_pipeline.py --max_steps 10 --max_samples 100

# Full training run
python examples/helpsteer2_complete_pipeline.py --max_steps 1000

# Skip training and only run inference on existing model
python examples/helpsteer2_complete_pipeline.py --skip_training

# Include benchmarking
python examples/helpsteer2_complete_pipeline.py --run_benchmarks
```

### Advanced Configuration

```bash
# Custom model and training parameters
python examples/helpsteer2_complete_pipeline.py \
    --model "microsoft/DialoGPT-medium" \
    --max_seq_length 4096 \
    --batch_size 4 \
    --max_steps 500 \
    --max_samples 5000 \
    --output_dir "./my_helpsteer2_model"
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `microsoft/DialoGPT-medium` | Base model to fine-tune |
| `--max_seq_length` | `2048` | Maximum sequence length |
| `--batch_size` | `2` | Training batch size per device |
| `--max_steps` | `100` | Maximum training steps |
| `--max_samples` | `None` | Limit dataset samples (None = all) |
| `--output_dir` | `./pantheraml_helpsteer2_model` | Output directory |
| `--skip_training` | `False` | Skip training, inference only |
| `--run_benchmarks` | `False` | Run benchmarks after training |

## Dataset Format

The nvidia/HelpSteer2 dataset contains helpful conversations with the following structure:

```python
{
    "prompt": "What are the benefits of regular exercise?",
    "response": "Regular exercise provides numerous benefits including..."
}
```

The script automatically formats these into chat templates:

```python
[
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response}
]
```

## Training Configuration

### LoRA Parameters
- **Rank (r)**: 16 (balance between performance and efficiency)
- **Alpha**: 16 (scaling factor)
- **Target modules**: All attention and MLP layers
- **Dropout**: 0 (optimized for speed)

### Training Arguments
- **Optimizer**: AdamW 8-bit (memory efficient)
- **Learning rate**: 2e-4 (conservative for stability)
- **Scheduler**: Linear decay with warmup
- **Precision**: BF16 (if supported), else FP16

## Output Formats

The script saves models in multiple formats:

### 1. LoRA Adapter
```
./pantheraml_helpsteer2_model/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files...
```

### 2. Merged Model (16-bit)
```
./pantheraml_helpsteer2_model_merged/
├── config.json
├── model.safetensors
└── tokenizer files...
```

### 3. GGUF Format (Optional)
```
./pantheraml_helpsteer2_model_gguf/
└── model.gguf
```

## Inference Examples

The script generates responses for these example prompts:

1. "What are the benefits of regular exercise?"
2. "How can I improve my time management skills?"
3. "Explain the concept of machine learning in simple terms."
4. "What are some effective study techniques for students?"
5. "How do I create a budget for personal finances?"

Results are saved to `inference_results.json`:

```json
[
    {
        "prompt": "What are the benefits of regular exercise?",
        "response": "Regular exercise offers numerous advantages..."
    }
]
```

## Using the Trained Model

### In Python Scripts

```python
from pantheraml import FastLanguageModel

# Load the trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./pantheraml_helpsteer2_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Format prompt
messages = [{"role": "user", "content": "Your question here"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

# Generate response
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
print(response)
```

### With PantheraML CLI

```bash
# Quick inference
pantheraml-cli --model ./pantheraml_helpsteer2_model --prompt "How do I learn Python?"

# Interactive chat
pantheraml-cli --model ./pantheraml_helpsteer2_model --interactive
```

## Performance Optimization

### Memory Usage
- **4-bit quantization**: Reduces memory by ~75%
- **LoRA**: Only trains ~1% of parameters
- **Gradient checkpointing**: Trades compute for memory

### Speed Optimization
- **Flash Attention**: Faster attention computation
- **Optimized kernels**: Custom CUDA kernels for common operations
- **Mixed precision**: FP16/BF16 training

### Multi-GPU Support
```bash
# Use multiple GPUs automatically
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/helpsteer2_complete_pipeline.py
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size and sequence length
   python examples/helpsteer2_complete_pipeline.py --batch_size 1 --max_seq_length 1024
   ```

2. **Slow training**
   ```bash
   # Enable gradient accumulation
   # Modify gradient_accumulation_steps in the script
   ```

3. **Poor results**
   ```bash
   # Increase training steps and data
   python examples/helpsteer2_complete_pipeline.py --max_steps 1000 --max_samples 10000
   ```

### Requirements

```bash
pip install torch transformers datasets trl accelerate bitsandbytes
```

For full PantheraML features:
```bash
pip install flash-attn triton ninja packaging
```

## Advanced Usage

### Custom Chat Templates

```python
# Modify the formatting function in the script
def formatting_prompts_func(examples):
    convos = []
    for prompt, response in zip(examples["prompt"], examples["response"]):
        # Custom format
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        convos.append(text)
    return {"text": convos}
```

### Custom Training Arguments

```python
# Modify TrainingArguments in train_model function
args=TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    max_steps=1000,
    learning_rate=1e-4,
    # Add your custom arguments here
)
```

### Integration with Other Datasets

The script can be adapted for other instruction-following datasets:

```python
# For different dataset formats
def format_custom_dataset(examples):
    # Adapt this function for your dataset structure
    convos = []
    for item in examples:
        # Format according to your data
        text = format_conversation(item)
        convos.append(text)
    return {"text": convos}
```

## Best Practices

1. **Start small**: Use `--max_samples 100 --max_steps 10` for testing
2. **Monitor memory**: Watch GPU memory usage during training
3. **Save frequently**: Use `save_steps=50` to avoid losing progress
4. **Validate results**: Always run inference examples after training
5. **Backup models**: Save in multiple formats for different use cases

## Support

For issues and questions:
- Check the PantheraML documentation
- Review the complete API reference
- Run the validation script: `python test_helpsteer2_pipeline.py`
