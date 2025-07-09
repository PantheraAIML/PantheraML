#!/usr/bin/env python3
"""
Simple test for device mapping utility functions
"""

import os
import sys

# Add the pantheraml path
sys.path.insert(0, "/Users/aayanmishra/unsloth")

def test_device_mapping_functions():
    """Test device mapping functions directly"""
    print("🧪 Testing device mapping utility functions...")
    
    # Test the utility functions directly without importing the full pantheraml module
    
    # Mock the DEVICE_TYPE for testing
    import pantheraml.models._utils as utils_module
    
    # Temporarily override the DEVICE_TYPE import
    original_device_type = getattr(utils_module, 'DEVICE_TYPE', 'cuda')
    
    # Test with TPU
    class MockPantheraML:
        DEVICE_TYPE = "tpu"
    
    # Temporarily patch the import
    import sys
    sys.modules['pantheraml'] = MockPantheraML()
    
    try:
        from pantheraml.models._utils import get_pytorch_device, get_autocast_device
        
        # Test device conversion
        pytorch_device = get_pytorch_device()
        pytorch_device_0 = get_pytorch_device(0)
        autocast_device = get_autocast_device()
        
        print(f"   PyTorch device (no ID): {pytorch_device}")
        print(f"   PyTorch device (ID=0): {pytorch_device_0}")
        print(f"   Autocast device: {autocast_device}")
        
        # Verify conversions
        assert pytorch_device == "xla", f"Expected 'xla', got '{pytorch_device}'"
        assert pytorch_device_0 == "xla:0", f"Expected 'xla:0', got '{pytorch_device_0}'"
        assert autocast_device == "cpu", f"Expected 'cpu', got '{autocast_device}'"
        
        print("✅ TPU device mapping working correctly!")
        
        # Test with CUDA
        MockPantheraML.DEVICE_TYPE = "cuda"
        
        pytorch_device = get_pytorch_device()
        pytorch_device_0 = get_pytorch_device(0)
        autocast_device = get_autocast_device()
        
        print(f"   CUDA PyTorch device (no ID): {pytorch_device}")
        print(f"   CUDA PyTorch device (ID=0): {pytorch_device_0}")
        print(f"   CUDA Autocast device: {autocast_device}")
        
        assert pytorch_device == "cuda", f"Expected 'cuda', got '{pytorch_device}'"
        assert pytorch_device_0 == "cuda:0", f"Expected 'cuda:0', got '{pytorch_device_0}'"
        assert autocast_device == "cuda", f"Expected 'cuda', got '{autocast_device}'"
        
        print("✅ CUDA device mapping working correctly!")
        
    except Exception as e:
        print(f"❌ Device mapping test failed: {e}")
        raise
    finally:
        # Clean up
        if 'pantheraml' in sys.modules:
            del sys.modules['pantheraml']

def test_torch_tensor_creation():
    """Test that we can create tensors with xla device string"""
    print("🧪 Testing PyTorch tensor creation with XLA device...")
    
    try:
        import torch
        
        # Test that XLA device string works (even if XLA isn't available)
        # This should not crash with "device type at start of device string" error
        
        try:
            # This may fail if XLA isn't available, but it shouldn't fail with device string parsing
            tensor = torch.tensor([1, 2, 3]).to("xla")
            print("✅ XLA device string accepted by PyTorch!")
        except Exception as e:
            if "Expected one of cpu, cuda" in str(e) and "device type at start of device string" in str(e):
                print("❌ PyTorch still has device string parsing issues")
                raise
            else:
                print(f"⚠️  XLA not available (expected): {e}")
                print("✅ But device string parsing works correctly!")
        
        # Test that regular device creation works
        tensor = torch.tensor([1, 2, 3]).to("cpu")
        print("✅ CPU tensor creation works!")
        
    except Exception as e:
        print(f"❌ Tensor creation test failed: {e}")
        raise

if __name__ == "__main__":
    print("🔧 Testing device mapping utility functions...")
    print("=" * 50)
    
    test_device_mapping_functions()
    print()
    test_torch_tensor_creation()
    
    print()
    print("🎉 All device mapping tests passed!")
