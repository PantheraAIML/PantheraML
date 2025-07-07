#!/usr/bin/env python3
"""
Test script for PantheraML multi-GPU functionality.

This script tests the basic multi-GPU setup and provides diagnostics.
"""

import torch
import sys
import os

def test_gpu_setup():
    """Test basic GPU setup and detection."""
    print("🔍 Testing GPU Setup")
    print("=" * 50)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA not available. Multi-GPU features require CUDA.")
        return False
    
    # Check number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    if num_gpus < 2:
        print("⚠️  Only 1 GPU detected. Multi-GPU features require 2+ GPUs.")
        print("   Testing will continue with single GPU simulation.")
    
    # Test each GPU
    for i in range(num_gpus):
        try:
            with torch.cuda.device(i):
                gpu_props = torch.cuda.get_device_properties(i)
                memory_gb = gpu_props.total_memory / (1024**3)
                print(f"GPU {i}: {gpu_props.name} ({memory_gb:.1f}GB)")
        except Exception as e:
            print(f"❌ Error accessing GPU {i}: {e}")
            return False
    
    print("✅ GPU setup looks good!")
    return True


def test_distributed_imports():
    """Test importing distributed components."""
    print("\n🔍 Testing Distributed Imports")
    print("=" * 50)
    
    try:
        from pantheraml.distributed import (
            MultiGPUConfig,
            setup_multi_gpu,
            cleanup_multi_gpu,
            is_distributed_available,
            get_world_size,
            get_rank,
            is_main_process,
        )
        print("✅ Successfully imported distributed components")
        
        # Test basic functions
        print(f"Distributed available: {is_distributed_available()}")
        print(f"World size: {get_world_size()}")
        print(f"Rank: {get_rank()}")
        print(f"Is main process: {is_main_process()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import distributed components: {e}")
        return False


def test_trainer_imports():
    """Test importing trainer components."""
    print("\n🔍 Testing Trainer Imports")
    print("=" * 50)
    
    try:
        from pantheraml import (
            PantheraMLDistributedTrainer,
            MultiGPUConfig,
        )
        print("✅ Successfully imported trainer components")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import trainer components: {e}")
        return False


def test_model_loading():
    """Test loading a model with multi-GPU support."""
    print("\n🔍 Testing Model Loading")
    print("=" * 50)
    
    try:
        from pantheraml import FastLanguageModel
        
        # Try to load a small model
        print("Loading model with multi-GPU support...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/tinyllama-chat-bnb-4bit",  # Small model for testing
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
            use_multi_gpu=torch.cuda.device_count() > 1,
            auto_device_map=True,
        )
        
        print("✅ Successfully loaded model with multi-GPU support")
        
        # Check model device placement
        print("\n📍 Model Device Placement:")
        device_map = {}
        for name, param in model.named_parameters():
            device = str(param.device)
            if device not in device_map:
                device_map[device] = []
            device_map[device].append(name)
        
        for device, params in device_map.items():
            print(f"   {device}: {len(params)} parameters")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False


def test_distributed_trainer():
    """Test creating a distributed trainer."""
    print("\n🔍 Testing Distributed Trainer")
    print("=" * 50)
    
    try:
        from pantheraml import (
            FastLanguageModel,
            UnslothDistributedTrainer,
            MultiGPUConfig,
        )
        from trl import SFTConfig
        
        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/tinyllama-chat-bnb-4bit",
            max_seq_length=256,
            dtype=None,
            load_in_4bit=True,
            use_multi_gpu=False,  # Keep simple for testing
        )
        
        # Prepare for training
        FastLanguageModel.for_training(model)
        
        # Create trainer config
        config = MultiGPUConfig(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            auto_device_map=True,
            use_gradient_checkpointing=True,
        )
        
        # Create minimal training args
        training_args = SFTConfig(
            per_device_train_batch_size=1,
            max_steps=1,
            output_dir="./test_output",
            logging_steps=1,
            save_strategy="no",
        )
        
        # Create trainer (without actual training)
        trainer = UnslothDistributedTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            multi_gpu_config=config,
            auto_setup_distributed=False,  # Don't actually setup distributed for test
        )
        
        print("✅ Successfully created UnslothDistributedTrainer")
        
        # Test memory stats
        memory_stats = trainer.get_model_memory_usage()
        if memory_stats:
            print("📊 Memory Stats:")
            for gpu, stats in memory_stats.items():
                print(f"   {gpu}: {stats['allocated']:.2f}GB allocated")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create distributed trainer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utilities():
    """Test utility functions."""
    print("\n🔍 Testing Utilities")
    print("=" * 50)
    
    try:
        from pantheraml.kernels.utils import (
            get_current_gpu_index,
            get_gpu_memory_info,
            clear_gpu_cache,
        )
        
        if torch.cuda.is_available():
            current_gpu = get_current_gpu_index()
            print(f"Current GPU index: {current_gpu}")
            
            memory_info = get_gpu_memory_info()
            print(f"GPU memory info: {memory_info}")
            
            # Test cache clearing
            clear_gpu_cache()
            print("✅ Successfully cleared GPU cache")
        
        print("✅ Utility functions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Failed utility tests: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Unsloth Multi-GPU Test Suite")
    print("=" * 60)
    
    tests = [
        ("GPU Setup", test_gpu_setup),
        ("Distributed Imports", test_distributed_imports),
        ("Trainer Imports", test_trainer_imports),
        ("Model Loading", test_model_loading),
        ("Distributed Trainer", test_distributed_trainer),
        ("Utilities", test_utilities),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Multi-GPU support is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
