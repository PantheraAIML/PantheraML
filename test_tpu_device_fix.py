#!/usr/bin/env python3
"""
Test TPU device string fix - verify that "tpu" is properly converted to "xla" for PyTorch
"""

import os
import sys

# Set up environment for TPU testing
os.environ["UNSLOTH_DEVICE_TYPE"] = "tpu"
os.environ["PANTHERAML_DEV_MODE"] = "1"  # Enable dev mode to bypass device checks

# Add the pantheraml path
sys.path.insert(0, "/Users/aayanmishra/unsloth")

def test_device_string_conversion():
    """Test that TPU device strings are properly converted to XLA"""
    print("🧪 Testing TPU device string conversion...")
    
    try:
        # Import the utility functions
        from pantheraml.models._utils import get_pytorch_device, get_autocast_device
        from pantheraml import DEVICE_TYPE
        
        print(f"   PantheraML DEVICE_TYPE: {DEVICE_TYPE}")
        
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
        
        print("✅ Device string conversion working correctly!")
        
    except Exception as e:
        print(f"❌ Device string conversion test failed: {e}")
        raise

def test_rotary_embedding_fix():
    """Test that rotary embedding can be initialized without device errors"""
    print("🧪 Testing rotary embedding initialization...")
    
    try:
        import torch
        from pantheraml.models.llama import LlamaRotaryEmbedding
        
        # Create a simple config-like object
        class MockConfig:
            rope_theta = 10000.0
            partial_rotary_factor = 1.0
            hidden_size = 4096
            num_attention_heads = 32
            max_position_embeddings = 4096
            
        config = MockConfig()
        
        # Try to create rotary embedding - this should not fail with device errors
        rope = LlamaRotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            config=config
        )
        
        print(f"✅ Rotary embedding created successfully!")
        print(f"   Current RoPE size: {rope.current_rope_size}")
        print(f"   Cos cache device: {rope.cos_cached.device}")
        print(f"   Sin cache device: {rope.sin_cached.device}")
        
    except Exception as e:
        print(f"❌ Rotary embedding test failed: {e}")
        raise

if __name__ == "__main__":
    print("🔧 Testing TPU device string fixes...")
    print("=" * 50)
    
    test_device_string_conversion()
    print()
    test_rotary_embedding_fix()
    
    print()
    print("🎉 All TPU device string tests passed!")
