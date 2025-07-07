"""
KAGGLE NOTEBOOK CELLS - Copy these directly into your Kaggle notebook
"""

# =============================================================================
# CELL 1: Setup and Install (if needed)
# =============================================================================
# Uncomment if you need to install packages
# !pip install transformers datasets trl accelerate bitsandbytes

import torch
print(f"🔥 CUDA available: {torch.cuda.is_available()}")
print(f"📊 GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# =============================================================================
# CELL 2: Import the Pipeline (paste the helpsteer2_complete_pipeline.py content here)
# =============================================================================
# Copy the entire helpsteer2_complete_pipeline.py file content into this cell
# Or use: exec(open('/kaggle/input/your-dataset/helpsteer2_complete_pipeline.py').read())

# =============================================================================
# CELL 3: Quick Test Run
# =============================================================================
print("🚀 Starting PantheraML HelpSteer2 Quick Test...")

# Quick test with minimal resources
results = kaggle_quick_test()

print("\n📊 Results Summary:")
print(f"✅ Training completed: {results['training_completed']}")
print(f"📁 Model saved to: {results['model_path']}")

if results.get('inference_results'):
    print(f"🔮 Generated {len(results['inference_results'])} inference examples")
    print("\n🎯 Sample outputs:")
    for i, result in enumerate(results['inference_results'][:2]):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {result['prompt']}")
        print(f"Response: {result['response'][:150]}...")

# =============================================================================
# CELL 4: Full Training (Optional)
# =============================================================================
# Uncomment for full training run
"""
print("🚀 Starting Full Training...")
results = kaggle_full_training()

print(f"✅ Full training completed: {results['training_completed']}")
if results.get('benchmark_results'):
    print(f"📊 Benchmark results: {results['benchmark_results']}")
"""

# =============================================================================
# CELL 5: Custom Configuration
# =============================================================================
print("🛠️ Running custom configuration...")

# Custom training for Kaggle GPU limits
results = run_kaggle_pipeline(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=1024,        # Reduced for memory
    batch_size=1,               # Conservative for Kaggle
    max_steps=50,               # Quick run
    max_samples=500,            # Limited dataset
    multi_gpu=True,             # Use both GPUs
    learning_rate=2e-4,
    output_dir="./qwen_helpsteer2_kaggle"
)

print(f"🎉 Custom training completed: {results['training_completed']}")

# =============================================================================
# CELL 6: Memory Monitoring
# =============================================================================
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")

print("💾 Initial GPU Memory:")
print_gpu_memory()

# =============================================================================
# CELL 7: Inference Testing
# =============================================================================
# Test inference with custom prompts
def test_custom_inference():
    # Load the trained model
    try:
        from pantheraml import FastLanguageModel
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="./qwen_helpsteer2_kaggle",  # Use your trained model
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        
        # Custom test prompts
        test_prompts = [
            "How do I prepare for a job interview?",
            "What's the best way to learn programming?",
            "Explain quantum computing in simple terms."
        ]
        
        print("🔮 Testing custom inference...")
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Custom Test {i+1} ---")
            print(f"Prompt: {prompt}")
            
            # Format and generate
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"❌ Custom inference failed: {e}")
        print("💡 Make sure the model was trained successfully in previous cells")

# Uncomment to run custom inference
# test_custom_inference()

# =============================================================================
# CELL 8: Save Results for Download
# =============================================================================
import json
import os

def save_training_artifacts():
    """Save all training artifacts for download"""
    
    artifacts = {
        "model_info": {
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "dataset": "nvidia/HelpSteer2",
            "training_framework": "PantheraML",
            "lora_rank": 16,
            "training_completed": True
        },
        "training_config": {
            "max_seq_length": 1024,
            "batch_size": 1,
            "learning_rate": 2e-4,
            "max_steps": 50,
            "multi_gpu": True
        },
        "files_created": []
    }
    
    # Check what files were created
    for path in ["./qwen_helpsteer2_kaggle", "./qwen_helpsteer2_kaggle_merged", "./qwen_helpsteer2_kaggle_gguf"]:
        if os.path.exists(path):
            artifacts["files_created"].append(path)
    
    # Save summary
    with open("training_summary.json", "w") as f:
        json.dump(artifacts, f, indent=2)
    
    print("📁 Training artifacts summary:")
    print(json.dumps(artifacts, indent=2))
    print("\n💾 Files ready for download:")
    print("- training_summary.json")
    print("- inference_results.json (if created)")
    for path in artifacts["files_created"]:
        print(f"- {path}/ (model directory)")

save_training_artifacts()

print("\n🎉 Kaggle pipeline completed! Check the output files above.")
print("💡 Use the model directories for deployment or further fine-tuning.")
