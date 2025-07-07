#!/usr/bin/env python3
"""
Basic functionality test for PantheraML.
This test checks that the core functionality works without requiring GPU hardware.
"""

import sys
import os

# Add the current directory to Python path so we can import unsloth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that basic imports work."""
    print("🧪 Testing PantheraML imports...")
    
    try:
        # These should work even without GPU
        from pantheraml.distributed import MultiGPUConfig, MultiTPUConfig
        print("✅ Distributed configs imported successfully")
        
        from pantheraml.kernels.utils import get_device_memory_info
        print("✅ Kernel utilities imported successfully")
        
        # Test benchmarking imports
        from pantheraml.benchmarks import (
            benchmark_mmlu, 
            PantheraBench, 
            BenchmarkResult,
            MMLUBenchmark
        )
        print("✅ Benchmarking utilities imported successfully")
        
        # Test the memory info function with CPU fallback
        memory_info = get_device_memory_info('cpu')
        print(f"✅ Device memory info (CPU fallback): {memory_info}")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_configs():
    """Test that configuration classes can be instantiated."""
    print("\n🧪 Testing configuration classes...")
    
    try:
        from pantheraml.distributed import MultiGPUConfig, MultiTPUConfig
        
        # Test MultiGPUConfig
        gpu_config = MultiGPUConfig(
            num_gpus=2,
            auto_device_map=True,
            use_deepspeed=False
        )
        print(f"✅ MultiGPUConfig created: {gpu_config.num_gpus} GPUs, auto_device_map={gpu_config.auto_device_map}")
        
        # Test MultiTPUConfig  
        tpu_config = MultiTPUConfig(
            num_cores=8,
            mesh_shape=(2, 4)
        )
        print(f"✅ MultiTPUConfig created: {tpu_config.num_cores} cores, mesh_shape={tpu_config.mesh_shape}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_syntax_validation():
    """Test that all core files have valid Python syntax."""
    print("\n🧪 Testing syntax validation...")
    
    import ast
    
    core_files = [
        'pantheraml/__init__.py',
        'pantheraml/distributed.py',
        'pantheraml/trainer.py', 
        'pantheraml/models/loader.py',
        'pantheraml/kernels/utils.py',
        'pantheraml/benchmarks.py'
    ]
    
    all_valid = True
    for filename in core_files:
        try:
            with open(filename, 'r') as f:
                source = f.read()
            ast.parse(source, filename=filename)
            print(f"✅ {filename} - valid syntax")
        except Exception as e:
            print(f"❌ {filename} - syntax error: {e}")
            all_valid = False
    
    return all_valid

def test_benchmarking_imports():
    """Test that benchmarking functions can be imported."""
    print("\n🧪 Testing benchmarking imports...")
    
    try:
        # Test individual benchmark function imports
        from pantheraml import benchmark_mmlu, benchmark_hellaswag, benchmark_arc
        print("✅ Individual benchmark functions imported successfully")
        
        # Test benchmark classes
        from pantheraml import PantheraBench, MMLUBenchmark, HellaSwagBenchmark, ARCBenchmark
        print("✅ Benchmark classes imported successfully")
        
        # Test benchmark result class
        from pantheraml import BenchmarkResult
        print("✅ BenchmarkResult class imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Benchmarking import test failed: {e}")
        return False

def test_pantherbench_instantiation():
    """Test that PantheraBench can be instantiated without actual model."""
    print("\n🧪 Testing PantheraBench instantiation...")
    
    try:
        # Create a mock model and tokenizer for testing
        class MockModel:
            def __init__(self):
                self.name_or_path = "test_model"
            
            def parameters(self):
                import torch
                return [torch.tensor([1.0])]
        
        class MockTokenizer:
            def __init__(self):
                self.eos_token_id = 0
        
        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()
        
        # Test PantheraBench instantiation
        from pantheraml import PantheraBench
        bench = PantheraBench(mock_model, mock_tokenizer)
        print("✅ PantheraBench instantiated successfully")
        
        # Check that methods exist
        assert hasattr(bench, 'mmlu'), "PantheraBench should have mmlu method"
        assert hasattr(bench, 'hellaswag'), "PantheraBench should have hellaswag method"
        assert hasattr(bench, 'arc_challenge'), "PantheraBench should have arc_challenge method"
        assert hasattr(bench, 'run_suite'), "PantheraBench should have run_suite method"
        print("✅ PantheraBench has all expected methods")
        
        return True
    except Exception as e:
        print(f"❌ PantheraBench instantiation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🦥 PantheraML Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configs), 
        ("Syntax Validation", test_syntax_validation),
        ("Benchmarking Imports Test", test_benchmarking_imports),
        ("PantheraBench Instantiation Test", test_pantherbench_instantiation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PantheraML is working correctly.")
        print("\n🙏 Credits:")
        print("   PantheraML extends the excellent work of the Unsloth team.")
        print("   Original Unsloth: https://github.com/unslothai/unsloth")
        print("\n🚀 New Features:")
        print("   • Multi-GPU distributed training support")
        print("   • 🧪 EXPERIMENTAL TPU support") 
        print("   • Enhanced memory optimization")
        print("   • Built-in benchmarking (MMLU, HellaSwag, ARC)")
        print("   • Export capabilities for results")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
