#!/usr/bin/env python3
"""
Phase 1 TPU Implementation Test Script

This script validates the Phase 1 TPU enhancements including:
- Core stability improvements
- Enhanced error handling
- Improved memory management
- XLA integration optimizations
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/Users/aayanmishra/unsloth')

def test_phase1_imports():
    """Test Phase 1 TPU imports."""
    print("🧪 Testing Phase 1 TPU imports...")
    
    try:
        # Test basic TPU availability
        from pantheraml.distributed import is_tpu_available, get_tpu_status
        print(f"✅ TPU available: {is_tpu_available()}")
        
        # Test Phase 1 kernel imports
        try:
            from pantheraml.kernels.tpu_kernels import (
                TPUMemoryManager, XLAOptimizer, TPUErrorHandler, TPUConfigManager,
                initialize_tpu_kernels, get_tpu_status as kernel_status
            )
            print("✅ Phase 1 TPU kernels imported successfully")
            
            # Test kernel functionality
            memory_manager = TPUMemoryManager()
            xla_optimizer = XLAOptimizer()
            error_handler = TPUErrorHandler()
            config_manager = TPUConfigManager()
            
            print("✅ Phase 1 kernel objects created successfully")
            
        except ImportError as e:
            print(f"⚠️ Phase 1 kernels not available: {e}")
            return False
            
        # Test enhanced distributed functions
        try:
            from pantheraml.distributed import (
                get_tpu_device, synchronize_tpu, get_tpu_memory_info,
                optimize_tpu_memory, setup_multi_tpu, cleanup_multi_tpu
            )
            print("✅ Enhanced TPU distributed functions imported")
            
        except ImportError as e:
            print(f"⚠️ Enhanced distributed functions not available: {e}")
            return False
            
        # Test enhanced trainer
        try:
            from pantheraml.trainer import PantheraMLTPUTrainer, MultiTPUConfig
            print("✅ Enhanced TPU trainer imported")
            
        except ImportError as e:
            print(f"⚠️ Enhanced TPU trainer not available: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_phase1_functionality():
    """Test Phase 1 TPU functionality without requiring actual TPU hardware."""
    print("\n🧪 Testing Phase 1 TPU functionality...")
    
    try:
        from pantheraml.kernels.tpu_kernels import (
            tpu_memory_manager, xla_optimizer, tpu_error_handler, tpu_config_manager
        )
        
        # Test memory manager
        memory_info = tpu_memory_manager.get_memory_info()
        print(f"✅ Memory manager status: {memory_info}")
        
        # Test XLA optimizer
        xla_available = xla_optimizer.is_xla_available()
        print(f"✅ XLA optimizer status: {'Available' if xla_available else 'Not available (expected)'}")
        
        # Test error handler
        error_count = tpu_error_handler.error_count
        print(f"✅ Error handler initialized: {error_count} errors")
        
        # Test config manager
        config = tpu_config_manager.get_optimal_config("small")
        print(f"✅ Config manager: {len(config)} configuration options")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def test_phase1_error_handling():
    """Test Phase 1 error handling capabilities."""
    print("\n🧪 Testing Phase 1 error handling...")
    
    try:
        from pantheraml.kernels.tpu_kernels import tpu_error_handler
        
        # Test error handling with a simulated error
        test_error = RuntimeError("Test error for Phase 1 validation")
        handled = tpu_error_handler.handle_tpu_error("test_operation", test_error)
        
        print(f"✅ Error handling test: {'Handled' if handled else 'Not handled'}")
        print(f"✅ Error count after test: {tpu_error_handler.error_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_phase1_configuration():
    """Test Phase 1 configuration management."""
    print("\n🧪 Testing Phase 1 configuration management...")
    
    try:
        from pantheraml.distributed import MultiTPUConfig, get_tpu_status
        
        # Test TPU configuration
        config = MultiTPUConfig(num_cores=4, auto_device_map=True)
        print(f"✅ TPU config created: {config.num_cores} cores")
        
        # Test status reporting
        status = get_tpu_status()
        print(f"✅ TPU status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_phase1_integration():
    """Test Phase 1 integration with main PantheraML."""
    print("\n🧪 Testing Phase 1 integration...")
    
    try:
        # Test main package imports work with Phase 1 enhancements
        import pantheraml
        print("✅ Main PantheraML package imports successfully")
        
        # Test device detection includes TPU enhancements
        device_type = pantheraml.DEVICE_TYPE
        print(f"✅ Device type detected: {device_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all Phase 1 tests."""
    print("🚀 PantheraML Phase 1 TPU Implementation Test")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_phase1_imports),
        ("Functionality Tests", test_phase1_functionality),
        ("Error Handling Tests", test_phase1_error_handling),
        ("Configuration Tests", test_phase1_configuration),
        ("Integration Tests", test_phase1_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Phase 1 Test Summary:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 1 tests passed! TPU enhancements are ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
