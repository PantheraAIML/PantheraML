#!/usr/bin/env python3
"""
Test TPU driver initialization fix for PantheraML.

This script tests the "0 active drivers ([]). There should only be one." error
and implements proper TPU/XLA device initialization.
"""

import torch
import sys
import os

def test_tpu_driver_initialization():
    """Test TPU driver initialization patterns"""
    print("Testing TPU driver initialization...")
    
    # Test current XLA availability
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        print("✓ PyTorch XLA imported successfully")
        
        # Check if TPU is available
        try:
            device_count = xm.xrt_world_size()
            print(f"✓ XLA device count: {device_count}")
        except Exception as e:
            print(f"✗ XLA device count error: {e}")
            
        # Test proper device initialization
        try:
            device = xm.xla_device()
            print(f"✓ XLA device: {device}")
            
            # Test tensor creation on XLA device
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
            print(f"✓ Tensor created on XLA device: {test_tensor.device}")
            
        except Exception as e:
            print(f"✗ XLA device initialization error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
    except ImportError:
        print("✗ PyTorch XLA not available")
        return False
    
    return True

def test_device_string_conversion():
    """Test device string handling for TPU"""
    print("\nTesting device string conversion...")
    
    # Mock the device mapping functions
    def get_pytorch_device(device_str):
        """Convert device string for PyTorch compatibility"""
        if device_str == "tpu":
            return "xla"
        return device_str
    
    def get_autocast_device(device_str):
        """Get autocast-compatible device string"""
        if device_str in ["tpu", "xla"]:
            return "cpu"  # XLA doesn't support autocast
        return device_str
    
    # Test conversions
    test_cases = ["cuda", "cuda:0", "cpu", "tpu", "xla"]
    
    for device in test_cases:
        pytorch_device = get_pytorch_device(device)
        autocast_device = get_autocast_device(device)
        print(f"  {device} -> PyTorch: {pytorch_device}, Autocast: {autocast_device}")

def test_proper_xla_initialization():
    """Test proper XLA initialization sequence"""
    print("\nTesting proper XLA initialization sequence...")
    
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        
        # Method 1: Use xm.xla_device() without arguments
        print("Method 1: xm.xla_device()")
        try:
            device = xm.xla_device()
            print(f"✓ Device: {device}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Method 2: Use ordinal specification
        print("Method 2: xm.xla_device(ordinal=0)")
        try:
            device = xm.xla_device(ordinal=0)
            print(f"✓ Device: {device}")
        except Exception as e:
            print(f"✗ Error: {e}")
            
        # Method 3: Check for TPU availability first
        print("Method 3: Check TPU availability")
        try:
            if xm.xrt_world_size() > 0:
                device = xm.xla_device()
                print(f"✓ TPU available, device: {device}")
            else:
                print("✗ No TPU devices available")
        except Exception as e:
            print(f"✗ Error checking TPU availability: {e}")
            
    except ImportError:
        print("✗ PyTorch XLA not available")

def create_tpu_safe_device_utils():
    """Create TPU-safe device utility functions"""
    print("\nCreating TPU-safe device utilities...")
    
    code = '''
def get_tpu_safe_device():
    """Get TPU device safely, handling driver initialization"""
    try:
        import torch_xla.core.xla_model as xm
        
        # Check if TPU is available first
        if xm.xrt_world_size() == 0:
            raise RuntimeError("No TPU devices available")
            
        # Get device with proper error handling
        device = xm.xla_device()
        return device
        
    except Exception as e:
        # Fallback to CPU if TPU fails
        print(f"TPU initialization failed: {e}")
        print("Falling back to CPU device")
        return torch.device("cpu")

def initialize_tpu_context():
    """Initialize TPU context properly"""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        
        # Force XLA initialization
        device = xm.xla_device()
        
        # Create a dummy tensor to initialize the device
        dummy = torch.tensor([1.0], device=device)
        xm.mark_step()  # Synchronize
        
        return device
        
    except Exception as e:
        print(f"TPU context initialization failed: {e}")
        return None

def safe_device_placement(tensor_or_model, target_device):
    """Safely place tensor or model on target device"""
    try:
        if target_device.type == "xla":
            # For XLA devices, ensure proper initialization
            import torch_xla.core.xla_model as xm
            result = tensor_or_model.to(target_device)
            xm.mark_step()  # Synchronize XLA operations
            return result
        else:
            return tensor_or_model.to(target_device)
            
    except Exception as e:
        print(f"Device placement failed: {e}")
        # Fallback to CPU
        return tensor_or_model.to("cpu")
'''
    
    print("✓ TPU-safe device utilities created")
    return code

if __name__ == "__main__":
    print("PantheraML TPU Driver Fix Test")
    print("=" * 50)
    
    # Run tests
    test_device_string_conversion()
    test_tpu_driver_initialization()
    test_proper_xla_initialization()
    
    # Create utilities
    utils_code = create_tpu_safe_device_utils()
    print("\nGenerated TPU-safe utilities:")
    print(utils_code)
    
    print("\n" + "=" * 50)
    print("TPU driver fix test completed")
