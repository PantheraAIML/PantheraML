#!/usr/bin/env python3
"""
Isolated test for TPU device utilities without importing full model system.
Tests only the specific utility functions we've added.
"""

import os
import sys
import importlib.util

# Force TPU device type for testing
os.environ["PANTHERAML_DEVICE_TYPE"] = "tpu"
os.environ["PANTHERAML_ALLOW_CPU"] = "1"

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_device_utilities_isolated():
    """Test TPU device utilities in isolation"""
    print("=" * 60)
    print("🧪 Testing TPU Device Utilities (Isolated)")
    print("=" * 60)
    
    try:
        # Test 1: Import device utility functions directly
        print("\n📦 Testing direct utility imports...")
        
        # We'll import the utility functions directly from the file
        # without going through the full pantheraml import system
        spec = importlib.util.spec_from_file_location(
            "utils", 
            "/Users/aayanmishra/unsloth/pantheraml/models/_utils.py"
        )
        utils_module = importlib.util.module_from_spec(spec)
        
        # Mock the problematic imports that require triton
        sys.modules['pantheraml_zoo'] = type(sys)('mock_zoo')
        sys.modules['pantheraml_zoo.utils'] = type(sys)('mock_zoo_utils')
        sys.modules['pantheraml_zoo.utils'].Version = type('Version', (), {'__init__': lambda self, x=None: None})
        sys.modules['pantheraml_zoo.patching_utils'] = type(sys)('mock_patching')
        for attr in ['patch_model_forward', 'patch_transformer_attention', 'patch_transformer_embeddings', 
                     'patch_tokenizer', 'patch_saving_functions', 'patch_training_functions']:
            setattr(sys.modules['pantheraml_zoo.patching_utils'], attr, lambda *args, **kwargs: None)
        
        spec.loader.exec_module(utils_module)
        
        print("✅ Successfully imported utility functions")
        
        # Test 2: Test device string conversion
        print("\n🔧 Testing device string conversion:")
        pytorch_device = utils_module.get_pytorch_device("tpu")
        print(f"  get_pytorch_device('tpu') = '{pytorch_device}'")
        assert pytorch_device == "xla", f"Expected 'xla', got '{pytorch_device}'"
        
        autocast_device = utils_module.get_autocast_device("tpu")
        print(f"  get_autocast_device('tpu') = '{autocast_device}'")
        assert autocast_device == "xla", f"Expected 'xla', got '{autocast_device}'"
        
        # Test 3: Test TPU safe device functions
        print("\n🛡️  Testing TPU safe device functions:")
        safe_device = utils_module.get_tpu_safe_device("tpu")
        print(f"  get_tpu_safe_device('tpu') = '{safe_device}'")
        assert safe_device == "xla", f"Expected 'xla', got '{safe_device}'"
        
        safe_placement = utils_module.safe_device_placement("tpu")
        print(f"  safe_device_placement('tpu') = '{safe_placement}'")
        assert safe_placement == "xla", f"Expected 'xla', got '{safe_placement}'"
        
        fallback_device = utils_module.get_device_with_fallback("tpu", "cpu")
        print(f"  get_device_with_fallback('tpu', 'cpu') = '{fallback_device}'")
        assert fallback_device in ["xla", "cpu"], f"Expected 'xla' or 'cpu', got '{fallback_device}'"
        
        # Test 4: Test inference context functions
        print("\n🔄 Testing inference context functions:")
        
        # Test get_inference_context
        with utils_module.get_inference_context("tpu") as ctx:
            print(f"  get_inference_context('tpu') context = {type(ctx).__name__}")
            # On TPU/XLA, should be torch.no_grad()
        
        with utils_module.tpu_compatible_inference_mode("tpu"):
            print("  tpu_compatible_inference_mode('tpu') context executed successfully")
        
        # Test 5: Test TPU initialization functions
        print("\n🚀 Testing TPU initialization functions:")
        
        # Test initialize_tpu_context
        result = utils_module.initialize_tpu_context()
        print(f"  initialize_tpu_context() = {result}")
        
        # Test ensure_tpu_initialization
        utils_module.ensure_tpu_initialization("tpu")
        print("  ensure_tpu_initialization('tpu') executed successfully")
        
        print("\n✅ All TPU device utilities passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ TPU device utilities failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_string_mappings():
    """Test all device string mappings"""
    print("\n" + "=" * 60)
    print("🗺️  Testing Device String Mappings")
    print("=" * 60)
    
    import importlib.util
    
    try:
        # Import utils module directly
        spec = importlib.util.spec_from_file_location(
            "utils", 
            "/Users/aayanmishra/unsloth/pantheraml/models/_utils.py"
        )
        utils_module = importlib.util.module_from_spec(spec)
        
        # Mock dependencies
        sys.modules['pantheraml_zoo'] = type(sys)('mock_zoo')
        sys.modules['pantheraml_zoo.utils'] = type(sys)('mock_zoo_utils')
        sys.modules['pantheraml_zoo.utils'].Version = type('Version', (), {'__init__': lambda self, x=None: None})
        sys.modules['pantheraml_zoo.patching_utils'] = type(sys)('mock_patching')
        for attr in ['patch_model_forward', 'patch_transformer_attention', 'patch_transformer_embeddings', 
                     'patch_tokenizer', 'patch_saving_functions', 'patch_training_functions']:
            setattr(sys.modules['pantheraml_zoo.patching_utils'], attr, lambda *args, **kwargs: None)
        
        spec.loader.exec_module(utils_module)
        
        # Test mappings for all device types
        test_cases = [
            ("cuda", "cuda", "cuda"),
            ("cuda:0", "cuda:0", "cuda"),
            ("xpu", "xpu", "xpu"),
            ("xpu:0", "xpu:0", "xpu"),
            ("tpu", "xla", "xla"),  # Key mapping: tpu -> xla for PyTorch
            ("cpu", "cpu", "cpu"),
        ]
        
        print("\nTesting device string mappings:")
        for input_device, expected_pytorch, expected_autocast in test_cases:
            pytorch_result = utils_module.get_pytorch_device(input_device)
            autocast_result = utils_module.get_autocast_device(input_device)
            
            print(f"  {input_device:8} -> pytorch: {pytorch_result:8} | autocast: {autocast_result:8}")
            
            assert pytorch_result == expected_pytorch, f"PyTorch mapping failed: {input_device} -> {pytorch_result}, expected {expected_pytorch}"
            assert autocast_result == expected_autocast, f"Autocast mapping failed: {input_device} -> {autocast_result}, expected {expected_autocast}"
        
        print("\n✅ All device string mappings passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Device string mappings failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all isolated tests"""
    print("TPU Device Utilities Isolated Test")
    print("=" * 60)
    
    import importlib.util
    
    tests = [
        test_device_utilities_isolated,
        test_device_string_mappings,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! TPU device utilities are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
