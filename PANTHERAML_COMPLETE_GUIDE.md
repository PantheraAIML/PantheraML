# PantheraML Complete Guide: All Use Cases & Examples

## Table of Contents
1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Basic Model Loading](#basic-model-loading)
4. [Training Examples](#training-examples)
5. [Saving & Loading Models](#saving--loading-models)
6. [Model Format Conversions](#model-format-conversions)
7. [Benchmarking & Evaluation](#benchmarking--evaluation)
8. [Multi-GPU & Distributed Training](#multi-gpu--distributed-training)
9. [TPU Support (Experimental)](#tpu-support-experimental)
10. [Advanced Use Cases](#advanced-use-cases)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

PantheraML is a high-performance machine learning framework designed for 2x faster training and inference of large language models. It provides seamless integration with Hugging Face transformers, optimized kernels, and support for various model formats including GGUF, GGML, and Ollama.

### Key Features
- 🚀 **2x Faster Training**: Optimized kernels and memory management
- 🔧 **Easy Integration**: Drop-in replacement for Hugging Face workflows
- 📊 **Built-in Benchmarking**: MMLU, HellaSwag, ARC, and more
- 🌐 **Multi-GPU Support**: Efficient distributed training
- 🔄 **Format Conversion**: GGUF, GGML, Ollama model formats
- 💾 **Memory Optimization**: Reduced memory usage with 4-bit quantization

---

## Installation & Setup

### Basic Installation
```bash
pip install pantheraml
```

### Development Installation
```bash
git clone https://github.com/PantheraML/pantheraml.git
cd pantheraml
pip install -e .
```

### Verify Installation
```python
import pantheraml
print(f"PantheraML version: {pantheraml.__version__}")
pantheraml.show_install_info()
```

---

## Basic Model Loading

### Loading a Base Model
```python
from pantheraml import FastLanguageModel
import torch

# Load a 7B Llama model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Enable training mode for fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

### Loading Different Model Types
```python
# Mistral 7B
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-v0.1",
    max_seq_length=4096,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Code Llama
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/codellama-7b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Gemma 2B
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

---

## Training Examples

### Basic Fine-tuning
```python
from pantheraml import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Prepare your dataset
dataset = [
    {"instruction": "What is machine learning?", 
     "output": "Machine learning is a subset of AI..."},
    {"instruction": "Explain neural networks", 
     "output": "Neural networks are computing systems..."},
]

# Format for training
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text)
    return {"text": texts}

# Set up trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

# Train the model
trainer.train()
```

### Chat Model Fine-tuning
```python
# Chat template formatting
def format_chat_template(examples):
    texts = []
    for conversation in examples["conversations"]:
        text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

# Apply formatting
dataset = dataset.map(format_chat_template, batched=True)

# Train with chat-optimized settings
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=100,
        learning_rate=1e-4,
        logging_steps=5,
        output_dir="chat_model_outputs",
        save_strategy="steps",
        save_steps=50,
    ),
)
```

### DPO (Direct Preference Optimization)
```python
from pantheraml import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer

# Load model for DPO
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path/to/your/sft/model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Patch for DPO compatibility
PatchDPOTrainer()

# Set up DPO trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # We'll use PEFT reference model
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=5e-7,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        seed=42,
        output_dir="dpo_outputs",
    ),
    beta=0.1,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    max_length=1024,
    max_prompt_length=512,
)

dpo_trainer.train()
```

---

## Saving & Loading Models

### Standard Saving Methods
```python
# Save LoRA adapters only
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Save merged 16-bit model
model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")

# Save merged 4-bit model (for continued training)
model.save_pretrained_merged("merged_4bit_model", tokenizer, save_method="merged_4bit")
```

### Push to Hugging Face Hub
```python
# Push LoRA adapters
model.push_to_hub("your_username/model_name", tokenizer=tokenizer, token="your_token")

# Push merged model
model.push_to_hub_merged(
    "your_username/merged_model", 
    tokenizer=tokenizer, 
    save_method="merged_16bit",
    token="your_token",
    private=False
)
```

### Advanced Saving Options
```python
from pantheraml.save import pantheraml_save_model

# Custom save with specific settings
pantheraml_save_model(
    model=model,
    tokenizer=tokenizer,
    save_directory="custom_model",
    save_method="merged_16bit",
    push_to_hub=False,
    max_shard_size="5GB",
    safe_serialization=True,
    commit_message="Trained with PantheraML",
    tags=["pantheraml", "llama", "fine-tuned"],
    maximum_memory_usage=0.8
)
```

---

## Model Format Conversions

### GGUF Conversion
```python
# Convert to GGUF format
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method="q4_k_m",  # Options: q8_0, q4_k_m, q5_k_m, etc.
    first_conversion="f16",
    push_to_hub=False
)

# Multiple quantization levels
model.save_pretrained_gguf(
    "model_gguf_multi",
    tokenizer,
    quantization_method=["q8_0", "q4_k_m", "q5_k_m"],
    push_to_hub=True,
    token="your_token"
)

# Available quantization methods
from pantheraml.save import print_quantization_methods
print_quantization_methods()
```

### GGML Conversion
```python
# Convert LoRA to GGML
model.save_pretrained_ggml("model_ggml", tokenizer)

# Push GGML to hub
model.push_to_hub_ggml(
    tokenizer=tokenizer,
    repo_id="your_username/model_ggml",
    token="your_token"
)
```

### Ollama Integration
```python
from pantheraml.save import push_to_ollama

# Convert and push to Ollama
gguf_location = "model.gguf"
push_to_ollama(
    tokenizer=tokenizer,
    gguf_location=gguf_location,
    username="your_username",
    model_name="my_model",
    tag="latest"
)
```

---

## Benchmarking & Evaluation

### Built-in Benchmarks
```python
from pantheraml import run_benchmark, BenchmarkRunner

# Run MMLU benchmark
results = run_benchmark(
    model_name="your_model_path",
    benchmark="mmlu",
    device="cuda",
    batch_size=4
)
print(f"MMLU Score: {results['accuracy']:.2%}")

# Run multiple benchmarks
runner = BenchmarkRunner(
    model_name="your_model_path",
    device="cuda"
)

# Available benchmarks
benchmarks = ["mmlu", "hellaswag", "arc_easy", "arc_challenge"]
all_results = {}

for benchmark in benchmarks:
    result = runner.run(benchmark, batch_size=4)
    all_results[benchmark] = result
    print(f"{benchmark}: {result['accuracy']:.2%}")

# Export results
runner.export_results("benchmark_results.json")
```

### Custom Evaluation
```python
def custom_evaluation(model, tokenizer, test_dataset):
    model.eval()
    correct = 0
    total = 0
    
    for example in test_dataset:
        inputs = tokenizer(example["input"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if example["expected"] in generated:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return {"accuracy": accuracy, "correct": correct, "total": total}

# Run evaluation
eval_results = custom_evaluation(model, tokenizer, test_data)
print(f"Custom evaluation accuracy: {eval_results['accuracy']:.2%}")
```

---

## Multi-GPU & Distributed Training

### Basic Multi-GPU Setup
```python
from pantheraml.distributed import setup_distributed_training
import torch.distributed as dist

# Initialize distributed training
setup_distributed_training()

# Load model on multiple GPUs
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="auto",  # Automatically distribute across GPUs
)

# Training with multiple GPUs
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        output_dir="multi_gpu_outputs",
    ),
)
```

### Advanced Distributed Training
```python
# Launch with torchrun
# torchrun --nproc_per_node=4 train_script.py

import os
from pantheraml.distributed import DistributedTrainingManager

def main():
    # Initialize distributed manager
    dist_manager = DistributedTrainingManager()
    dist_manager.setup()
    
    # Load model with proper device mapping
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-2-13b-bnb-4bit",
        max_seq_length=4096,
        load_in_4bit=True,
        device_map={"": dist_manager.local_rank},
    )
    
    # Configure for distributed training
    model = dist_manager.prepare_model(model)
    
    # Training arguments for distributed setup
    training_args = TrainingArguments(
        output_dir="distributed_outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=10,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    # Save only on main process
    if dist_manager.is_main_process():
        model.save_pretrained_merged("final_model", tokenizer)
    
    dist_manager.cleanup()

if __name__ == "__main__":
    main()
```

---

## TPU Support (Experimental)

### Basic TPU Setup
```python
from pantheraml.tpu import setup_tpu_training
import torch_xla.core.xla_model as xm

# Initialize TPU
setup_tpu_training()

# Load model for TPU
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b",
    max_seq_length=2048,
    dtype=torch.bfloat16,  # TPUs work best with bfloat16
    load_in_4bit=False,    # 4-bit not supported on TPU
)

# TPU-optimized training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        bf16=True,  # Use bfloat16 on TPU
        dataloader_num_workers=0,  # TPU works better with 0 workers
        output_dir="tpu_outputs",
    ),
)

trainer.train()
```

### TPU Monitoring
```python
from pantheraml.tpu import TPUMonitor

# Monitor TPU utilization
monitor = TPUMonitor()

# During training
for step in range(num_steps):
    # Training step
    loss = trainer.training_step(model, inputs)
    
    # Log TPU metrics
    if step % 10 == 0:
        metrics = monitor.get_metrics()
        print(f"Step {step}: TPU Utilization: {metrics['utilization']:.1%}")
```

---

## Advanced Use Cases

### Custom LoRA Configurations
```python
# Advanced LoRA setup
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Higher rank for more capacity
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",  # Include embeddings
    ],
    lora_alpha=32,
    lora_dropout=0.1,
    bias="lora_only",  # Add bias to LoRA layers
    task_type="CAUSAL_LM",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

### Memory Optimization
```python
from pantheraml.kernels.utils import optimize_memory_usage

# Optimize memory for large models
optimize_memory_usage(
    model=model,
    max_memory_usage=0.85,  # Use 85% of available memory
    enable_gradient_checkpointing=True,
    use_fast_kernels=True,
)

# Manual memory management
import gc
import torch

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Call between training phases
clear_memory()
```

### Custom Tokenization
```python
# Add custom tokens
new_tokens = ["<custom_token>", "<special_instruction>"]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# Custom chat template
chat_template = """{% for message in messages %}
{{'<|' + message['role'] + '|>\n' + message['content'] + '<|end|>\n'}}
{% endfor %}
{% if add_generation_prompt %}
{{'<|assistant|>\n'}}
{% endif %}"""

tokenizer.chat_template = chat_template
```

### Model Merging
```python
from pantheraml.models.merger import merge_lora_weights

# Merge LoRA weights back into base model
merged_model = merge_lora_weights(
    base_model=base_model,
    lora_model=lora_model,
    merge_ratio=0.8,  # 80% LoRA, 20% base
)

# Save merged model
merged_model.save_pretrained("merged_custom_model")
```

### Inference Optimization
```python
# Fast inference setup
FastLanguageModel.for_inference(model)

# Streaming generation
def generate_stream(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.multinomial(
                torch.softmax(next_token_logits, dim=-1), 
                num_samples=1
            )
            
            # Yield the new token
            yield tokenizer.decode(next_token, skip_special_tokens=True)
            
            # Update inputs for next iteration
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token.unsqueeze(0)], dim=1)
            inputs["attention_mask"] = torch.cat([
                inputs["attention_mask"], 
                torch.ones(1, 1, device="cuda")
            ], dim=1)

# Use streaming generation
for token in generate_stream("Explain quantum computing"):
    print(token, end="", flush=True)
```

---

## Troubleshooting

### Common Issues and Solutions

#### CUDA Out of Memory
```python
# Reduce batch size and increase gradient accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Reduce from 2 to 1
    gradient_accumulation_steps=8,  # Increase to maintain effective batch size
    dataloader_pin_memory=False,    # Reduce memory usage
    max_grad_norm=1.0,             # Prevent gradient explosion
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

#### Slow Training
```python
# Optimize training speed
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    dataloader_num_workers=4,      # Parallel data loading
    fp16=True,                     # Use mixed precision
    optim="adamw_torch_fused",     # Faster optimizer
    group_by_length=True,          # Group similar length sequences
)
```

#### Model Loading Issues
```python
# Debug model loading
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-2-7b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,  # For custom models
    )
except Exception as e:
    print(f"Loading failed: {e}")
    # Try without quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-2-7b",
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=False,
    )
```

### Performance Monitoring
```python
import time
import psutil
import torch

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.step_times = []
    
    def log_step(self, step, loss):
        current_time = time.time()
        step_time = current_time - self.start_time
        self.step_times.append(step_time)
        
        # Memory usage
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        cpu_memory = psutil.virtual_memory().percent
        
        print(f"Step {step}: Loss={loss:.4f}, "
              f"Time={step_time:.2f}s, "
              f"GPU={gpu_memory:.1f}GB, "
              f"CPU={cpu_memory:.1f}%")
        
        self.start_time = current_time
    
    def get_avg_step_time(self):
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0

# Use during training
monitor = PerformanceMonitor()
# In training loop:
# monitor.log_step(step, loss.item())
```

---

## Complete Example: End-to-End Fine-tuning

```python
#!/usr/bin/env python3
"""
Complete PantheraML Fine-tuning Example
This script demonstrates a full workflow from data preparation to model deployment.
"""

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from pantheraml import FastLanguageModel
from pantheraml.save import pantheraml_save_model
import json

def main():
    # 1. Load and prepare model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-2-7b-chat-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # 2. Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # 3. Prepare dataset
    dataset_dict = {
        "instruction": [
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
        ],
        "output": [
            "The capital of France is Paris.",
            "Machine learning is a way for computers to learn patterns from data without being explicitly programmed.",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        ]
    }
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = f"<s>[INST] {instruction} [/INST] {output} </s>"
            texts.append(text)
        return {"text": texts}
    
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # 4. Set up training
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=10,  # Short for demo
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )
    
    # 5. Train
    print("Starting training...")
    trainer.train()
    
    # 6. Save model
    print("Saving model...")
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    
    # 7. Save merged model
    model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")
    
    # 8. Convert to GGUF
    print("Converting to GGUF...")
    model.save_pretrained_gguf(
        "gguf_model",
        tokenizer,
        quantization_method="q4_k_m",
        push_to_hub=False
    )
    
    # 9. Test inference
    print("Testing inference...")
    FastLanguageModel.for_inference(model)
    
    inputs = tokenizer(
        "<s>[INST] What is 2+2? [/INST]",
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model response: {response}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
```

---

## CLI Usage

PantheraML also provides a command-line interface for common operations:

```bash
# Basic training
pantheraml-cli train --model unsloth/llama-2-7b-chat --dataset data.json --output my_model

# Convert model formats
pantheraml-cli convert --input my_model --output my_model.gguf --format gguf --quant q4_k_m

# Run benchmarks
pantheraml-cli benchmark --model my_model --benchmarks mmlu,hellaswag --output results.json

# Multi-GPU training
pantheraml-cli train --model unsloth/llama-2-13b --dataset large_data.json --gpus 4 --output large_model

# Push to Hugging Face Hub
pantheraml-cli push --model my_model --repo username/model_name --token $HF_TOKEN
```

---

This comprehensive guide covers all major use cases of PantheraML. For the latest updates and additional examples, visit the [PantheraML GitHub repository](https://github.com/PantheraML/pantheraml).

**Happy training with PantheraML! 🚀**
