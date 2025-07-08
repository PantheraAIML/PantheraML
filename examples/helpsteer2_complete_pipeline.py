#!/usr/bin/env python3
"""
Complete PantheraML Pipeline: Model Loading, Training, and Inference
Dataset: nvidia/HelpSteer2 (prompt -> response)

This script demonstrates the full workflow:
1. Loading a pre-trained model with PantheraML optimizations
2. Preparing the nvidia/HelpSteer2 dataset
3. Fine-tuning the model
4. Saving the trained model
5. Loading for inference
6. Running inference examples

Usage:
    python helpsteer2_complete_pipeline.py [--model MODEL_NAME] [--max_seq_length LENGTH] [--batch_size SIZE]
"""

import argparse
import os
import json

# Import PantheraML first for optimal performance
try:
    from pantheraml import FastLanguageModel
    from pantheraml.chat_templates import get_chat_template
    from pantheraml.trainer import pantheraml_train
    
    # Try to import PantheraMLTrainer, fallback to regular SFTTrainer
    try:
        from pantheraml.trainer import PantheraMLTrainer
        PANTHERAML_TRAINER_AVAILABLE = True
        print("✅ Successfully imported PantheraML components with PantheraMLTrainer")
    except ImportError:
        PANTHERAML_TRAINER_AVAILABLE = False
        print("✅ Successfully imported PantheraML components (using standard SFTTrainer)")
        
except ImportError as e:
    print(f"❌ Error importing PantheraML: {e}")
    print("Please ensure PantheraML is properly installed")
    PANTHERAML_TRAINER_AVAILABLE = False

# Now import other ML libraries
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# PantheraML Multi-GPU support - Define fallback functions first
def _fallback_setup_multi_gpu(*args, **kwargs):
    pass

def _fallback_is_multi_gpu_available():
    return False

def _fallback_get_world_size():
    return 1

def _fallback_get_rank():
    return 0

def _fallback_is_main_process():
    return True

def _fallback_cleanup_distributed():
    pass

# Try to import PantheraML distributed functions, use fallbacks if not available
try:
    from pantheraml.distributed import (
        setup_multi_gpu as _pantheraml_setup_multi_gpu, 
        is_multi_gpu_available,
        get_world_size,
        get_rank,
        is_main_process,
        cleanup_distributed
    )
    PANTHERAML_DISTRIBUTED_AVAILABLE = True
    print("✅ PantheraML distributed training available")
except ImportError:
    PANTHERAML_DISTRIBUTED_AVAILABLE = False
    print("⚠️ PantheraML distributed training not available")
    
    # Use fallback functions for single-GPU mode
    _pantheraml_setup_multi_gpu = _fallback_setup_multi_gpu
    is_multi_gpu_available = _fallback_is_multi_gpu_available
    get_world_size = _fallback_get_world_size
    get_rank = _fallback_get_rank
    is_main_process = _fallback_is_main_process
    cleanup_distributed = _fallback_cleanup_distributed

def setup_multi_gpu():
    """
    Setup multi-GPU training environment using PantheraML
    """
    if not PANTHERAML_DISTRIBUTED_AVAILABLE:
        print("⚠️ PantheraML distributed training not available")
        return False
    
    if torch.cuda.device_count() > 1:
        print(f"🚀 Multi-GPU detected: {torch.cuda.device_count()} GPUs available")
        
        try:
            # Initialize PantheraML multi-GPU setup
            success = _pantheraml_setup_multi_gpu()
            if success:
                print(f"✅ PantheraML multi-GPU training initialized")
                print(f"📊 World size: {get_world_size()}")
                print(f"🔢 Current rank: {get_rank()}")
                print(f"👑 Is main process: {is_main_process()}")
                return True
            else:
                print("⚠️ Failed to setup PantheraML multi-GPU training")
                print("🔄 Falling back to single GPU training")
                return False
        except Exception as e:
            print(f"⚠️ Failed to setup PantheraML multi-GPU training: {e}")
            print("🔄 Falling back to single GPU training")
            return False
    else:
        print(f"📱 Single GPU detected: {torch.cuda.device_count()} GPU")
        return False

def setup_model_and_tokenizer(model_name="Qwen/Qwen2.5-0.5B-Instruct", 
                              max_seq_length=2048, 
                              dtype=None, 
                              load_in_4bit=True):
    """
    Load and setup the model and tokenizer with PantheraML optimizations
    """
    print(f"🔄 Loading model: {model_name}")
    print(f"📏 Max sequence length: {max_seq_length}")
    print(f"💾 4-bit loading: {load_in_4bit}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # trust_remote_code=False, # Set to True for custom models
    )
    
    # Add LoRA adapters for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Supports any value, but = 0 is optimized
        bias="none",     # Supports any value, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # Must use "unsloth" (not "pantheraml") for compatibility with pantheraml_zoo
        random_state=3407,
        use_rslora=False,   # Support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    
    print("✅ Model and tokenizer loaded successfully")
    return model, tokenizer

def prepare_dataset(tokenizer, max_samples=None):
    """
    Load and prepare the nvidia/HelpSteer2 dataset
    """
    print("🔄 Loading nvidia/HelpSteer2 dataset...")
    
    # Load the dataset
    dataset = load_dataset("nvidia/HelpSteer2", split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"📊 Using {len(dataset)} samples for training")
    else:
        print(f"📊 Total samples in dataset: {len(dataset)}")
    
    # Define the chat template compatible with Qwen models
    chat_template = get_chat_template(
        tokenizer,
        chat_template="chatml",  # Qwen models typically use ChatML format
    )
    
    def formatting_prompts_func(examples):
        """
        Format the HelpSteer2 data for training
        """
        convos = []
        for prompt, response in zip(examples["prompt"], examples["response"]):
            # Create conversation format
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            convos.append(text)
        
        return {"text": convos}
    
    # Apply formatting
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print("✅ Dataset prepared successfully")
    return dataset

def train_model(model, tokenizer, dataset, output_dir="./pantheraml_helpsteer2_model", 
                per_device_train_batch_size=2, gradient_accumulation_steps=4, 
                warmup_steps=5, max_steps=100, learning_rate=2e-4, use_multi_gpu=False):
    """
    Train the model using PantheraML's optimized trainer with multi-GPU support
    """
    print("🚀 Starting model training...")
    
    # Setup multi-GPU if requested and available
    if use_multi_gpu and PANTHERAML_DISTRIBUTED_AVAILABLE:
        multi_gpu_setup = setup_multi_gpu()
        if multi_gpu_setup:
            print("✅ Using PantheraML multi-GPU training")
            # Adjust batch size for multi-GPU
            effective_batch_size = per_device_train_batch_size * get_world_size()
            print(f"📊 Effective batch size across {get_world_size()} GPUs: {effective_batch_size}")
        else:
            print("⚠️ Multi-GPU setup failed, using single GPU")
    else:
        print("📱 Using single GPU training")
    
    # Training arguments with multi-GPU support
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        dataloader_pin_memory=False,
        # Multi-GPU specific settings
        ddp_find_unused_parameters=False if use_multi_gpu else None,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    # Create trainer with appropriate class based on availability
    if PANTHERAML_TRAINER_AVAILABLE:
        print("🔧 Using PantheraMLTrainer for optimized training")
        trainer = PantheraMLTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            args=training_args,
        )
    else:
        print("🔧 Using standard SFTTrainer")
        # For compatibility with modified SFTTrainer, try without tokenizer first
        try:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=2048,
                dataset_num_proc=2,
                args=training_args,
            )
        except TypeError as e:
            if "unexpected keyword argument 'tokenizer'" in str(e):
                print("⚠️ SFTTrainer doesn't accept tokenizer parameter, trying without...")
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    dataset_text_field="text",
                    max_seq_length=2048,
                    dataset_num_proc=2,
                    args=training_args,
                )
            else:
                raise e
    
    # Show current memory stats
    if is_main_process() or not use_multi_gpu:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"💾 GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"💾 {start_gpu_memory} GB of memory reserved.")
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Show final memory and time stats
    if is_main_process() or not use_multi_gpu:
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
        print(f"💾 Peak reserved memory = {used_memory} GB.")
        print(f"💾 Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"💾 Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"💾 Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    print("✅ Training completed successfully!")
    return trainer

def save_trained_model(model, tokenizer, output_dir="./pantheraml_helpsteer2_model"):
    """
    Save the trained model in multiple formats (only on main process for multi-GPU)
    """
    # Only save on main process in multi-GPU setup
    if not is_main_process() and PANTHERAML_DISTRIBUTED_AVAILABLE:
        print(f"⏳ Rank {get_rank()}: Waiting for main process to save model...")
        return
    
    print("💾 Saving trained model...")
    
    # Save LoRA model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ LoRA model saved to: {output_dir}")
    
    # Save merged model (optional, for deployment)
    merged_output_dir = f"{output_dir}_merged"
    model.save_pretrained_merged(merged_output_dir, tokenizer, save_method="merged_16bit")
    print(f"✅ Merged model saved to: {merged_output_dir}")
    
    # Save as GGUF for efficient inference (optional)
    try:
        gguf_output_dir = f"{output_dir}_gguf"
        model.save_pretrained_gguf(gguf_output_dir, tokenizer)
        print(f"✅ GGUF model saved to: {gguf_output_dir}")
    except Exception as e:
        print(f"⚠️ GGUF save failed (optional): {e}")

def load_trained_model_for_inference(model_path="./pantheraml_helpsteer2_model"):
    """
    Load the trained model for inference
    """
    print(f"🔄 Loading trained model for inference from: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable faster inference
    FastLanguageModel.for_inference(model)
    
    print("✅ Model loaded for inference")
    return model, tokenizer

def run_inference_examples(model, tokenizer):
    """
    Run inference examples using the trained model
    """
    print("🔮 Running inference examples...")
    
    # Example prompts similar to HelpSteer2 format
    test_prompts = [
        "What are the benefits of regular exercise?",
        "How can I improve my time management skills?",
        "Explain the concept of machine learning in simple terms.",
        "What are some effective study techniques for students?",
        "How do I create a budget for personal finances?"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {prompt}")
        
        # Format the prompt
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(f"Response: {response}")
        
        results.append({
            "prompt": prompt,
            "response": response
        })
    
    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Inference examples completed! Results saved to 'inference_results.json'")
    return results

def benchmark_model(model, tokenizer):
    """
    Optional: Run benchmarks on the trained model
    """
    try:
        from pantheraml.benchmarks import run_benchmark
        print("🏃 Running model benchmarks...")
        
        # Run a quick benchmark
        results = run_benchmark(
            model=model,
            tokenizer=tokenizer,
            benchmark_type="mmlu",
            num_samples=100,  # Small sample for demo
            device="cuda"
        )
        
        print("📊 Benchmark Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
            
    except ImportError:
        print("⚠️ Benchmarking module not available, skipping...")
    except Exception as e:
        print(f"⚠️ Benchmarking failed: {e}")

def check_system_requirements():
    """
    Check system requirements and diagnose potential issues
    """
    print("🔍 Checking system requirements...")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ CUDA not available - PantheraML requires NVIDIA GPUs")
            return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False
    
    # Check available memory
    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 4:
            print(f"⚠️ Low GPU memory: {gpu_memory:.1f} GB (recommend 8+ GB)")
        else:
            print(f"✅ GPU memory: {gpu_memory:.1f} GB")
    except Exception as e:
        print(f"⚠️ Could not check GPU memory: {e}")
    
    # Check internet connectivity for dataset download
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            print("✅ Internet connectivity to HuggingFace available")
        else:
            print("⚠️ HuggingFace may not be accessible")
    except Exception as e:
        print(f"⚠️ Internet connectivity check failed: {e}")
    
    # Check HuggingFace datasets
    try:
        from datasets import load_dataset
        print("✅ HuggingFace datasets library available")
    except ImportError as e:
        print(f"❌ HuggingFace datasets not available: {e}")
        return False
    
    # Check transformers and trl
    try:
        from transformers import TrainingArguments
        from trl import SFTTrainer
        print("✅ Transformers and TRL libraries available")
    except ImportError as e:
        print(f"❌ Missing ML libraries: {e}")
        return False
    
    return True

def main():
    """
    Main function to run the complete pipeline
    """
    parser = argparse.ArgumentParser(description="PantheraML Complete Pipeline for nvidia/HelpSteer2")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", 
                       help="Base model to fine-tune")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size per device")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="Maximum training steps")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to use from dataset (None for all)")
    parser.add_argument("--output_dir", default="./pantheraml_helpsteer2_model",
                       help="Output directory for trained model")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training and only run inference")
    parser.add_argument("--run_benchmarks", action="store_true",
                       help="Run benchmarks after training")
    parser.add_argument("--multi_gpu", action="store_true",
                       help="Enable PantheraML multi-GPU training")
    
    args = parser.parse_args()
    
    print("🎯 PantheraML Complete Pipeline for nvidia/HelpSteer2")
    print("=" * 60)
    
    # Check multi-GPU availability and setup
    if args.multi_gpu:
        if PANTHERAML_DISTRIBUTED_AVAILABLE:
            print("🚀 Multi-GPU training requested and available")
        else:
            print("⚠️ Multi-GPU training requested but PantheraML distributed module not available")
            print("🔄 Falling back to single GPU training")
            args.multi_gpu = False
    
    # Check system requirements
    system_check_passed = check_system_requirements()
    
    if not args.skip_training and system_check_passed:
        # Step 1: Load model and tokenizer
        print("\n📋 Step 1: Loading model and tokenizer")
        model, tokenizer = setup_model_and_tokenizer(
            model_name=args.model,
            max_seq_length=args.max_seq_length
        )
        
        # Step 2: Prepare dataset
        print("\n📋 Step 2: Preparing dataset")
        dataset = prepare_dataset(tokenizer, max_samples=args.max_samples)
        
        # Step 3: Train model
        print("\n📋 Step 3: Training model")
        trainer = train_model(
            model, tokenizer, dataset,
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            max_steps=args.max_steps,
            use_multi_gpu=args.multi_gpu
        )
        
        # Step 4: Save model
        print("\n📋 Step 4: Saving trained model")
        save_trained_model(model, tokenizer, args.output_dir)
        
        # Optional: Run benchmarks
        if args.run_benchmarks:
            print("\n📋 Step 5: Running benchmarks")
            benchmark_model(model, tokenizer)
    
    # Step 5/6: Load for inference (only on main process for multi-GPU)
    if is_main_process() or not args.multi_gpu:
        print("\n📋 Step 5: Loading model for inference")
        model, tokenizer = load_trained_model_for_inference(args.output_dir)
        
        # Step 6/7: Run inference examples
        print("\n📋 Step 6: Running inference examples")
        results = run_inference_examples(model, tokenizer)
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"📁 Model saved in: {args.output_dir}")
        print(f"📄 Inference results: inference_results.json")
        print("\nTo use your trained model:")
        print(f"  from pantheraml import FastLanguageModel")
        print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{args.output_dir}')")
    
    # Cleanup distributed training
    if args.multi_gpu and PANTHERAML_DISTRIBUTED_AVAILABLE:
        try:
            cleanup_distributed()
            print("✅ Multi-GPU training cleanup completed")
        except Exception as e:
            print(f"⚠️ Multi-GPU cleanup warning: {e}")

def run_kaggle_pipeline(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=2048,
    batch_size=2,
    max_steps=100,
    max_samples=None,
    output_dir="./pantheraml_helpsteer2_model",
    skip_training=False,
    run_benchmarks=False,
    multi_gpu=True,  # Enable multi-GPU by default for Kaggle
    learning_rate=2e-4,
    gradient_accumulation_steps=4,
    warmup_steps=5
):
    """
    Kaggle-friendly function to run the complete pipeline without CLI
    
    Args:
        model_name: HuggingFace model name to fine-tune
        max_seq_length: Maximum sequence length for the model
        batch_size: Training batch size per device
        max_steps: Maximum training steps
        max_samples: Maximum samples to use from dataset (None for all)
        output_dir: Output directory for trained model
        skip_training: Skip training and only run inference
        run_benchmarks: Run benchmarks after training
        multi_gpu: Enable PantheraML multi-GPU training
        learning_rate: Learning rate for training
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_steps: Warmup steps for learning rate scheduler
    
    Returns:
        dict: Results containing model paths and inference results
    """
    
    print("🎯 PantheraML Complete Pipeline for nvidia/HelpSteer2 (Kaggle Mode)")
    print("=" * 70)
    
    # Check system requirements first
    if not check_system_requirements():
        print("❌ System requirements not met - training may fail")
        print("💡 This is likely why training is failing")
        results = {
            "model_path": output_dir,
            "merged_model_path": f"{output_dir}_merged", 
            "gguf_model_path": f"{output_dir}_gguf",
            "inference_results": None,
            "benchmark_results": None,
            "training_completed": False,
            "system_requirements_error": "System requirements not met"
        }
        return results
    
    print(f"📋 Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max steps: {max_steps}")
    print(f"  Max samples: {max_samples}")
    print(f"  Multi-GPU: {multi_gpu}")
    print(f"  Skip training: {skip_training}")
    print("=" * 70)
    
    results = {
        "model_path": output_dir,
        "merged_model_path": f"{output_dir}_merged",
        "gguf_model_path": f"{output_dir}_gguf",
        "inference_results": None,
        "benchmark_results": None,
        "training_completed": False
    }
    
    # Check multi-GPU availability and setup
    if multi_gpu:
        if PANTHERAML_DISTRIBUTED_AVAILABLE:
            print("🚀 Multi-GPU training requested and available")
        else:
            print("⚠️ Multi-GPU training requested but PantheraML distributed module not available")
            print("🔄 Falling back to single GPU training")
            multi_gpu = False
    
    # Check system requirements
    system_check_passed = check_system_requirements()
    
    if not skip_training and system_check_passed:
        try:
            # Step 1: Load model and tokenizer
            print("\n📋 Step 1: Loading model and tokenizer")
            print(f"🔄 Attempting to load model: {model_name}")
            
            try:
                model, tokenizer = setup_model_and_tokenizer(
                    model_name=model_name,
                    max_seq_length=max_seq_length
                )
                print("✅ Model and tokenizer loaded successfully")
            except Exception as model_error:
                print(f"❌ Model loading failed: {model_error}")
                raise model_error
            
            # Step 2: Prepare dataset
            print("\n📋 Step 2: Preparing dataset")
            print("🔄 Attempting to load nvidia/HelpSteer2 dataset...")
            
            try:
                dataset = prepare_dataset(tokenizer, max_samples=max_samples)
                print("✅ Dataset prepared successfully")
            except Exception as dataset_error:
                print(f"❌ Dataset preparation failed: {dataset_error}")
                raise dataset_error
            
            # Step 3: Train model
            print("\n📋 Step 3: Training model")
            print("🔄 Attempting to start training...")
            
            try:
                trainer = train_model(
                    model, tokenizer, dataset,
                    output_dir=output_dir,
                    per_device_train_batch_size=batch_size,
                    max_steps=max_steps,
                    learning_rate=learning_rate,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_steps=warmup_steps,
                    use_multi_gpu=multi_gpu
                )
                print("✅ Training completed successfully")
            except Exception as training_error:
                print(f"❌ Training failed: {training_error}")
                raise training_error
            
            # Step 4: Save model
            print("\n📋 Step 4: Saving trained model")
            save_trained_model(model, tokenizer, output_dir)
            results["training_completed"] = True
            
            # Optional: Run benchmarks
            if run_benchmarks:
                print("\n📋 Step 5: Running benchmarks")
                benchmark_results = benchmark_model(model, tokenizer)
                results["benchmark_results"] = benchmark_results
                
        except Exception as e:
            import traceback
            error_msg = str(e) if str(e) else "Unknown error"
            print(f"❌ Training failed: {error_msg}")
            print(f"📋 Full error details:")
            traceback.print_exc()
            print("🔄 Continuing with inference if model exists...")
            results["training_error"] = error_msg
            results["training_traceback"] = traceback.format_exc()
    
    # Step 5/6: Load for inference (only on main process for multi-GPU)
    try:
        if is_main_process() or not multi_gpu:
            print("\n📋 Step 5: Loading model for inference")
            
            # Try to load the trained model, fallback to base model if needed
            try:
                model, tokenizer = load_trained_model_for_inference(output_dir)
            except:
                print("⚠️ Trained model not found, loading base model for inference...")
                model, tokenizer = setup_model_and_tokenizer(
                    model_name=model_name,
                    max_seq_length=max_seq_length
                )
                FastLanguageModel.for_inference(model)
            
            # Step 6/7: Run inference examples
            print("\n📋 Step 6: Running inference examples")
            inference_results = run_inference_examples(model, tokenizer)
            results["inference_results"] = inference_results
            
            print("\n🎉 Pipeline completed successfully!")
            print(f"📁 Model saved in: {output_dir}")
            print(f"📄 Inference results saved to: inference_results.json")
            
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else "Unknown error"
        print(f"❌ Inference failed: {error_msg}")
        print(f"📋 Full error details:")
        traceback.print_exc()
        results["inference_error"] = error_msg
        results["inference_traceback"] = traceback.format_exc()
    
    # Cleanup distributed training
    if multi_gpu and PANTHERAML_DISTRIBUTED_AVAILABLE:
        try:
            cleanup_distributed()
            print("✅ Multi-GPU training cleanup completed")
        except Exception as e:
            print(f"⚠️ Multi-GPU cleanup warning: {e}")
    
    return results

# Kaggle usage examples
def kaggle_quick_test():
    """Quick test run for Kaggle (small dataset, few steps)"""
    return run_kaggle_pipeline(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_steps=10,
        max_samples=100,
        batch_size=1,  # Conservative for memory
        multi_gpu=True
    )

def kaggle_full_training():
    """Full training run for Kaggle"""
    return run_kaggle_pipeline(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_steps=500,
        max_samples=5000,
        batch_size=2,
        multi_gpu=True,
        run_benchmarks=True
    )

def kaggle_inference_only():
    """Inference only (assumes model already trained)"""
    return run_kaggle_pipeline(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        skip_training=True,
        multi_gpu=False  # Not needed for inference
    )

if __name__ == "__main__":
    # Check if running in Kaggle environment
    if os.path.exists('/kaggle'):
        print("🏠 Kaggle environment detected!")
        print("\n🚀 Running quick test...")
        print("To run full training, use: kaggle_full_training()")
        print("To run inference only, use: kaggle_inference_only()")
        
        # Run quick test by default in Kaggle
        results = kaggle_quick_test()
        print(f"\n📊 Results: {results}")
    else:
        # Use CLI mode for non-Kaggle environments
        main()
\