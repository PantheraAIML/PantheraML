#!/usr/bin/env python3
"""
🦥 PantheraML CLI Standalone Entry Point

This script sets development mode and then imports the CLI.
"""

import os
import sys

# Set development mode BEFORE any PantheraML imports
os.environ["PANTHERAML_DEV_MODE"] = "1"

def main():
    """Main CLI entry point with graceful error handling."""
    import argparse
    
    # Check if this is a help request
    if "--help" in sys.argv or "-h" in sys.argv:
        # Show help without importing PantheraML
        show_help()
        return
    
    try:
        # Import the actual CLI implementation
        from pantheraml.cli import main as cli_main
        cli_main()
    except ImportError as e:
        print("❌ Import Error")
        print("🚫 Failed to import PantheraML CLI components")
        print(f"Error: {e}")
        print("\n💡 Make sure PantheraML and its dependencies are properly installed:")
        print("   pip install -e .")
        print("   # For CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   pip install triton bitsandbytes")
        sys.exit(1)
    except NotImplementedError as e:
        if "PantheraML currently only works on NVIDIA GPUs" in str(e):
            print("❌ Device Compatibility Error")
            print("🚫 PantheraML requires NVIDIA GPUs, Intel GPUs, or TPUs")
            print("💡 Current system is not supported")
            print("\n🖥️  Supported devices:")
            print("   • NVIDIA GPUs (CUDA)")
            print("   • Intel GPUs")  
            print("   • TPUs (experimental)")
            print(f"\nError details: {e}")
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

def show_help():
    """Show help information without importing PantheraML dependencies."""
    help_text = """🦥 PantheraML CLI - Enhanced LLM Fine-tuning
Based on the excellent work of the Unsloth team

usage: pantheraml-cli [-h] [--model_name MODEL_NAME] [--max_seq_length MAX_SEQ_LENGTH] 
                     [--dtype DTYPE] [--load_in_4bit] [--dataset DATASET] [--r R] 
                     [--lora_alpha LORA_ALPHA] [--lora_dropout LORA_DROPOUT] 
                     [--bias BIAS] [--use_gradient_checkpointing USE_GRADIENT_CHECKPOINTING] 
                     [--random_state RANDOM_STATE] [--use_rslora] 
                     [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE] 
                     [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] 
                     [--warmup_steps WARMUP_STEPS] [--max_steps MAX_STEPS] 
                     [--learning_rate LEARNING_RATE] [--logging_steps LOGGING_STEPS] 
                     [--optim OPTIM] [--weight_decay WEIGHT_DECAY] 
                     [--lr_scheduler_type LR_SCHEDULER_TYPE] [--seed SEED] 
                     [--output_dir OUTPUT_DIR] [--report_to REPORT_TO] 
                     [--save_model] [--save_path SAVE_PATH] 
                     [--quantization_method QUANTIZATION_METHOD] [--push_model] 
                     [--hub_path HUB_PATH] [--hub_token HUB_TOKEN] [--benchmark] 
                     [--suite {all,mmlu,hellaswag,arc,custom}] [--samples SAMPLES] [--export]

🤖 Model Options:
  --model_name MODEL_NAME
                        Model name to load (default: unsloth/llama-3-8b)
  --max_seq_length MAX_SEQ_LENGTH
                        Maximum sequence length, default is 2048. We auto support RoPE Scaling internally!
  --dtype DTYPE         Data type for model (None for auto detection)
  --load_in_4bit        Use 4bit quantization to reduce memory usage
  --dataset DATASET     Huggingface dataset to use for training (default: yahma/alpaca-cleaned)

🧠 LoRA Options:
These options are used to configure the LoRA model.

  --r R                 Rank for Lora model, default is 16. (common values: 8, 16, 32, 64, 128)
  --lora_alpha LORA_ALPHA
                        LoRA alpha parameter, default is 16. (common values: 8, 16, 32, 64, 128)
  --lora_dropout LORA_DROPOUT
                        LoRA dropout rate, default is 0.0 which is optimized.
  --bias BIAS           Bias type for LoRA (default: none)
  --use_gradient_checkpointing USE_GRADIENT_CHECKPOINTING
                        Use gradient checkpointing (default: unsloth)
  --random_state RANDOM_STATE
                        Random state for reproducibility (default: 3407)
  --use_rslora          Use RSLoRA for more stable training

🏃 Training Options:
These options are used to configure the training process.

  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per device (default: 2)
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Gradient accumulation steps (default: 4)
  --warmup_steps WARMUP_STEPS
                        Number of warmup steps (default: 5)
  --max_steps MAX_STEPS
                        Maximum training steps (default: 60)
  --learning_rate LEARNING_RATE
                        Learning rate (default: 2e-4)
  --logging_steps LOGGING_STEPS
                        Log every N steps (default: 1)
  --optim OPTIM         Optimizer to use (default: adamw_8bit)
  --weight_decay WEIGHT_DECAY
                        Weight decay (default: 0.01)
  --lr_scheduler_type LR_SCHEDULER_TYPE
                        Learning rate scheduler (default: linear)
  --seed SEED           Random seed (default: 3407)
  --output_dir OUTPUT_DIR
                        Output directory (default: ./outputs)
  --report_to REPORT_TO
                        Reporting tool (default: tensorboard)

💾 Saving Options:
These options are used to configure model saving and pushing.

  --save_model          Save the trained model
  --save_path SAVE_PATH
                        Path to save the model (default: ./model)
  --quantization_method QUANTIZATION_METHOD
                        Quantization method for saving (default: f16)
  --push_model          Push model to Hugging Face Hub
  --hub_path HUB_PATH   Hugging Face Hub model path
  --hub_token HUB_TOKEN
                        Hugging Face Hub token

🧪 Benchmarking Options:
Run model benchmarks instead of training.

  --benchmark           Run benchmarking instead of training
  --suite {all,mmlu,hellaswag,arc,custom}
                        Benchmark suite to run (default: all)
  --samples SAMPLES     Maximum number of samples per benchmark (default: all)
  --export              Export benchmark results to JSON/CSV

🙏 Credits: 
   PantheraML extends the excellent work of the Unsloth team.
   Original Unsloth: https://github.com/unslothai/unsloth
   
🚀 PantheraML Enhancements:
   • Multi-GPU distributed training support  
   • 🧪 EXPERIMENTAL TPU support (requires torch_xla)
   • Enhanced memory optimization
   • Extended model compatibility
   
Happy fine-tuning! 🎯

⚠️  NOTE: PantheraML requires NVIDIA GPUs, Intel GPUs, or TPUs for training.
          Current system may not be supported for actual training."""
    print(help_text)

if __name__ == "__main__":
    main()
