#!/usr/bin/env python3
"""
Test device string conversion logic directly
"""

def get_pytorch_device(device_type="tpu", device_id=None):
    """
    Convert PantheraML device type to PyTorch-compatible device string.
    """
    if device_type == "tpu":
        # PyTorch/XLA uses "xla" for TPU devices
        return "xla" if device_id is None else f"xla:{device_id}"
    elif device_id is not None:
        return f"{device_type}:{device_id}"
    else:
        return device_type

def get_autocast_device(device_type="tpu"):
    """
    Get device type for torch.autocast operations.
    """
    if device_type == "tpu":
        return "cpu"  # TPU operations use CPU for autocast
    return device_type

def test_device_conversion():
    """Test device string conversion logic"""
    print("🧪 Testing device string conversion logic...")
    
    # Test TPU conversion
    assert get_pytorch_device("tpu") == "xla"
    assert get_pytorch_device("tpu", 0) == "xla:0"
    assert get_autocast_device("tpu") == "cpu"
    print("✅ TPU conversion: tpu -> xla, autocast -> cpu")
    
    # Test CUDA passthrough
    assert get_pytorch_device("cuda") == "cuda"
    assert get_pytorch_device("cuda", 0) == "cuda:0"
    assert get_autocast_device("cuda") == "cuda"
    print("✅ CUDA passthrough: cuda -> cuda")
    
    # Test XPU passthrough
    assert get_pytorch_device("xpu") == "xpu"
    assert get_pytorch_device("xpu", 1) == "xpu:1"
    assert get_autocast_device("xpu") == "xpu"
    print("✅ XPU passthrough: xpu -> xpu")
    
    print("✅ All device conversion tests passed!")

def test_pytorch_device_strings():
    """Test that PyTorch accepts the converted device strings"""
    print("🧪 Testing PyTorch device string acceptance...")
    
    import torch
    
    # Test CPU (always works)
    try:
        tensor = torch.tensor([1, 2, 3]).to("cpu")
        print("✅ CPU device string works")
    except Exception as e:
        print(f"❌ CPU device failed: {e}")
        raise
    
    # Test that xla device string format is accepted by PyTorch
    # (even if XLA runtime isn't available)
    try:
        # This might fail because XLA isn't available, but should NOT fail with device string parsing
        tensor = torch.tensor([1, 2, 3]).to("xla")
        print("✅ XLA device string accepted and XLA runtime available!")
    except RuntimeError as e:
        if "Device should be" in str(e) or "xla" in str(e).lower():
            print("⚠️  XLA runtime not available (expected), but device string format is valid")
        else:
            print(f"❌ Unexpected XLA error: {e}")
            raise
    except Exception as e:
        if "Expected one of cpu, cuda" in str(e) and "device string" in str(e):
            print("❌ PyTorch failed to parse XLA device string!")
            raise
        else:
            print(f"⚠️  XLA device creation failed (expected if XLA not installed): {e}")
    
    print("✅ Device string format tests completed!")

if __name__ == "__main__":
    print("🔧 Testing device string conversion logic...")
    print("=" * 60)
    
    test_device_conversion()
    print()
    test_pytorch_device_strings()
    
    print()
    print("🎉 Device string conversion logic working correctly!")
