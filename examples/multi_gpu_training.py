#!/usr/bin/env python3
"""
Multi-GPU Training Example with PantheraML

This script demonstrates how to use PantheraML's new multi-GPU support for faster training.
It shows both automatic setup and manual configuration options.

Usage:
    # Single node, multi-GPU (automatic)
    python multi_gpu_example.py
    
    # Multi-node setup using torchrun
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.100" multi_gpu_example.py
    
    # SLURM environment
    srun --nodes=2 --gpus-per-node=4 python multi_gpu_example.py
"""

import torch
from pantheraml import (
    FastLanguageModel, 
    PantheraMLDistributedTrainer,
    MultiGPUConfig,
    setup_multi_gpu,
    is_distributed_available,
    get_world_size,
    is_main_process,
)
from datasets import load_dataset
from trl import SFTConfig
import os


def main():
    """Main training function with multi-GPU support."""
    
    # Check multi-GPU availability
    if is_main_process():
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"🚀 Unsloth Multi-GPU Training Example")
        print(f"   Available GPUs: {num_gpus}")
        print(f"   Distributed available: {is_distributed_available()}")
        print(f"   World size: {get_world_size()}")
    
    # Configure multi-GPU settings
    multi_gpu_config = MultiGPUConfig(
        backend="nccl",  # Use NCCL for NVIDIA GPUs
        auto_device_map=True,
        use_gradient_checkpointing=True,
        find_unused_parameters=False,  # Set to True if you have unused parameters
    )
    
    # Model configuration
    model_name = "unsloth/llama-2-7b-chat-bnb-4bit"
    max_seq_length = 2048
    
    # Load model with automatic multi-GPU support
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
        use_multi_gpu=True,  # Enable multi-GPU support
        auto_device_map=True,  # Let Unsloth handle device mapping
        # max_memory={0: "10GB", 1: "10GB"},  # Optional: specify memory limits per GPU
    )
    
    # Enable training mode
    FastLanguageModel.for_training(model)
    
    # Load and prepare dataset
    if is_main_process():
        print("📊 Loading dataset...")
    
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")  # Small subset for demo
    
    def formatting_prompts_func(examples):
        """Format examples for training."""
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = f"### Instruction:\n{instruction}\n"
            if input_text:
                text += f"### Input:\n{input_text}\n"
            text += f"### Response:\n{output}"
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # Training configuration
    training_args = SFTConfig(
        per_device_train_batch_size=1,  # Adjust based on GPU memory
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
        output_dir="./outputs",
        save_steps=30,
        save_total_limit=2,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        
        # Multi-GPU specific settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        # auto_scale_lr=True,  # Uncomment to automatically scale learning rate by world size
    )
    
    # Create the distributed trainer
    trainer = PantheraMLDistributedTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=training_args,
        multi_gpu_config=multi_gpu_config,
        auto_setup_distributed=True,  # Automatically setup distributed training
    )
    
    # Print memory stats before training
    if is_main_process():
        trainer.print_memory_stats()
    
    # Start training
    if is_main_process():
        print("\n🏋️ Starting multi-GPU training...")
    
    trainer_stats = trainer.train()
    
    # Print final stats
    if is_main_process():
        print(f"\n✅ Training completed!")
        print(f"   Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
        print(f"   Samples per second: {trainer_stats.metrics['train_samples_per_second']:.2f}")
        print(f"   Steps per second: {trainer_stats.metrics['train_steps_per_second']:.2f}")
        
        # Print final memory stats
        trainer.print_memory_stats()
    
    # Save the model (only on main process)
    if is_main_process():
        print("\n💾 Saving model...")
        model.save_pretrained("./multi_gpu_lora_model")
        tokenizer.save_pretrained("./multi_gpu_lora_model")
        print("   Model saved to ./multi_gpu_lora_model")


def advanced_multi_gpu_example():
    """
    Advanced example showing manual multi-GPU setup and configuration.
    """
    if is_main_process():
        print("\n🔧 Advanced Multi-GPU Configuration Example")
    
    # Manual multi-GPU setup with custom configuration
    config = MultiGPUConfig(
        backend="nccl",
        timeout_minutes=60,  # Longer timeout for large models
        auto_device_map=True,
        tensor_parallel=False,  # Set to True for very large models
        pipeline_parallel=False,
        use_gradient_checkpointing=True,
        find_unused_parameters=True,  # Useful for complex model architectures
    )
    
    # Setup distributed environment
    setup_multi_gpu(config)
    
    # Manual device mapping example
    num_gpus = torch.cuda.device_count()
    max_memory = {i: "10GB" for i in range(num_gpus)}  # 10GB per GPU
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-2-7b-chat-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="balanced",  # Balance layers across GPUs
        max_memory=max_memory,
        use_multi_gpu=True,
    )
    
    if is_main_process():
        print(f"   Model distributed across {num_gpus} GPUs")
        print(f"   Memory limit per GPU: 10GB")
        
        # Print model device mapping
        print("\n📍 Model Device Mapping:")
        for name, param in model.named_parameters():
            if hasattr(param, 'device'):
                print(f"   {name}: {param.device}")


if __name__ == "__main__":
    # Check for command line arguments
    import sys
    
    if "--advanced" in sys.argv:
        advanced_multi_gpu_example()
    else:
        main()
