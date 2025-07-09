#!/usr/bin/env python3
"""
Comprehensive test for TPU driver initialization fixes in PantheraML.

This script tests all the new TPU safety functions and device initialization logic.
"""

import os
import sys
import torch

# Set device type for testing
os.environ["PANTHERAML_DEVICE_TYPE"] = "tpu"

def test_device_utility_imports():
    """Test importing device utility functions"""
    print("Testing device utility imports...")
    
    try:
        # Test importing from pantheraml models
        sys.path.insert(0, "/Users/aayanmishra/unsloth")
        from pantheraml.models._utils import (
            get_pytorch_device,
            get_autocast_device,
            get_inference_context,
            tpu_compatible_inference_mode,
            get_tpu_safe_device,
            initialize_tpu_context,
            safe_device_placement,
            get_device_with_fallback,
            ensure_tpu_initialization
        )
        
        print("✓ All device utility functions imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_device_string_conversion():
    """Test device string conversion functions"""
    print("\nTesting device string conversion...")
    
    try:
        from pantheraml.models._utils import get_pytorch_device, get_autocast_device
        
        # Test basic conversion
        pytorch_device = get_pytorch_device()
        autocast_device = get_autocast_device()
        
        print(f"  Default device: {pytorch_device}")
        print(f"  Autocast device: {autocast_device}")
        
        # Test with device ID
        pytorch_device_0 = get_pytorch_device(0)
        print(f"  Device 0: {pytorch_device_0}")
        
        # Verify TPU conversions
        if pytorch_device == "xla" and autocast_device == "cpu":
            print("✓ TPU device string conversion working correctly")
            return True
        else:
            print(f"✗ Unexpected device strings: {pytorch_device}, {autocast_device}")
            return False
            
    except Exception as e:
        print(f"✗ Device conversion test failed: {e}")
        return False

def test_safe_device_functions():
    """Test TPU-safe device functions"""
    print("\nTesting TPU-safe device functions...")
    
    try:
        from pantheraml.models._utils import (
            get_tpu_safe_device,
            get_device_with_fallback,
            ensure_tpu_initialization
        )
        
        # Test safe device retrieval
        print("  Testing get_tpu_safe_device...")
        safe_device = get_tpu_safe_device()
        print(f"    Safe device: {safe_device}")
        
        # Test device with fallback
        print("  Testing get_device_with_fallback...")
        fallback_device = get_device_with_fallback()
        print(f"    Fallback device: {fallback_device}")
        
        # Test TPU initialization check
        print("  Testing ensure_tpu_initialization...")
        tpu_available = ensure_tpu_initialization()
        print(f"    TPU available: {tpu_available}")
        
        print("✓ Safe device functions tested successfully")
        return True
        
    except Exception as e:
        print(f"✗ Safe device functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_context():
    """Test TPU-compatible inference context"""
    print("\nTesting inference context...")
    
    try:
        from pantheraml.models._utils import get_inference_context, tpu_compatible_inference_mode
        
        # Test inference context function
        print("  Testing get_inference_context...")
        context = get_inference_context()
        print(f"    Context type: {type(context).__name__}")
        
        # Test inference context usage
        with context:
            dummy_tensor = torch.tensor([1.0, 2.0, 3.0])
            result = dummy_tensor * 2
            print(f"    Tensor operation successful: {result}")
        
        # Test decorator
        print("  Testing tpu_compatible_inference_mode decorator...")
        
        @tpu_compatible_inference_mode
        def test_function():
            return torch.tensor([4.0, 5.0, 6.0]) * 3
        
        result = test_function()
        print(f"    Decorator function result: {result}")
        
        print("✓ Inference context tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Inference context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_placement():
    """Test safe device placement function"""
    print("\nTesting device placement...")
    
    try:
        from pantheraml.models._utils import safe_device_placement, get_tpu_safe_device
        
        # Create test tensor
        test_tensor = torch.tensor([1.0, 2.0, 3.0])
        print(f"  Original tensor device: {test_tensor.device}")
        
        # Get safe device
        target_device = get_tpu_safe_device()
        print(f"  Target device: {target_device}")
        
        # Test safe placement
        placed_tensor = safe_device_placement(test_tensor, target_device)
        print(f"  Placed tensor device: {placed_tensor.device}")
        
        print("✓ Device placement test passed")
        return True
        
    except Exception as e:
        print(f"✗ Device placement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_workflow():
    """Test a comprehensive TPU workflow simulation"""
    print("\nTesting comprehensive TPU workflow...")
    
    try:
        from pantheraml.models._utils import (
            ensure_tpu_initialization,
            get_device_with_fallback,
            safe_device_placement,
            tpu_compatible_inference_mode
        )
        
        # Step 1: Ensure TPU initialization
        print("  Step 1: Checking TPU initialization...")
        tpu_ready = ensure_tpu_initialization()
        print(f"    TPU ready: {tpu_ready}")
        
        # Step 2: Get safe device
        print("  Step 2: Getting safe device...")
        device = get_device_with_fallback()
        print(f"    Selected device: {device}")
        
        # Step 3: Create and place model components
        print("  Step 3: Creating model components...")
        
        @tpu_compatible_inference_mode
        def simulate_inference():
            # Simulate model weights
            weights = torch.randn(10, 10)
            input_data = torch.randn(5, 10)
            
            # Place on device safely
            weights = safe_device_placement(weights, device)
            input_data = safe_device_placement(input_data, device)
            
            # Simulate forward pass
            output = torch.matmul(input_data, weights.T)
            
            return output
        
        # Step 4: Run inference
        print("  Step 4: Running inference simulation...")
        result = simulate_inference()
        print(f"    Inference result shape: {result.shape}")
        print(f"    Result device: {result.device}")
        
        print("✓ Comprehensive workflow test passed")
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all TPU driver fix tests"""
    print("PantheraML TPU Driver Fix Comprehensive Test")
    print("=" * 50)
    
    tests = [
        test_device_utility_imports,
        test_device_string_conversion,
        test_safe_device_functions,
        test_inference_context,
        test_device_placement,
        test_comprehensive_workflow,
    ]
    
    results = []
    for test in tests:
        try:
            success = test()
            results.append(success)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"✓ Passed: {sum(results)}")
    print(f"✗ Failed: {len(results) - sum(results)}")
    print(f"Total: {len(results)}")
    
    if all(results):
        print("\n🎉 All TPU driver fix tests passed!")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
