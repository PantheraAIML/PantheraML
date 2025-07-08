#!/usr/bin/env python3
"""
Test script to verify that the torch_amp_custom_fwd import issue is fixed
"""

import os

# Set development mode to allow running on unsupported devices
os.environ["PANTHERAML_DEV_MODE"] = "1"

def test_torch_amp_import():
    """Test that torch_amp_custom_fwd and torch_amp_custom_bwd can be imported"""
    
    print("🧪 Testing torch_amp_custom_fwd import fix...")
    print("=" * 45)
    
    try:
        # Test import of the problematic function
        from pantheraml.models._utils import torch_amp_custom_fwd, torch_amp_custom_bwd
        
        print("✅ Successfully imported torch_amp_custom_fwd")
        print("✅ Successfully imported torch_amp_custom_bwd")
        
        # Test that they're callable
        if callable(torch_amp_custom_fwd):
            print("✅ torch_amp_custom_fwd is callable")
        else:
            print("❌ torch_amp_custom_fwd is not callable")
            return False
            
        if callable(torch_amp_custom_bwd):
            print("✅ torch_amp_custom_bwd is callable")
        else:
            print("❌ torch_amp_custom_bwd is not callable")
            return False
        
        # Test basic usage
        @torch_amp_custom_fwd
        def dummy_forward(x):
            return x * 2
        
        @torch_amp_custom_bwd
        def dummy_backward(x):
            return x / 2
        
        test_val = 5
        forward_result = dummy_forward(test_val)
        backward_result = dummy_backward(forward_result)
        
        if backward_result == test_val:
            print("✅ torch_amp functions work correctly")
        else:
            print(f"❌ torch_amp functions failed: {test_val} -> {forward_result} -> {backward_result}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_device_type_handling():
    """Test that different device types are handled correctly"""
    
    print("\n🔧 Testing device type handling...")
    print("=" * 35)
    
    try:
        from pantheraml import DEVICE_TYPE
        print(f"Current DEVICE_TYPE: {DEVICE_TYPE}")
        
        # Test import works regardless of device type
        from pantheraml.models._utils import torch_amp_custom_fwd, torch_amp_custom_bwd
        
        print(f"✅ torch_amp functions available for {DEVICE_TYPE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Device type handling failed: {e}")
        return False

def test_pantheraml_import():
    """Test that PantheraML can be imported without AttributeError"""
    
    print("\n📦 Testing full PantheraML import...")
    print("=" * 35)
    
    try:
        import pantheraml
        print("✅ PantheraML imported successfully")
        
        # Test that FastLanguageModel can be imported
        from pantheraml import FastLanguageModel
        print("✅ FastLanguageModel imported successfully")
        
        return True
        
    except AttributeError as e:
        if "torch_amp_custom_fwd" in str(e):
            print(f"❌ torch_amp_custom_fwd issue still exists: {e}")
            return False
        else:
            print(f"❌ Different AttributeError: {e}")
            return False
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Run all import tests"""
    
    print("🧪 PantheraML torch_amp Import Fix Test")
    print("=" * 40)
    print("Testing fix for AttributeError: torch_amp_custom_fwd not found")
    print()
    
    tests = [
        ("torch_amp Import", test_torch_amp_import),
        ("Device Type Handling", test_device_type_handling),
        ("Full PantheraML Import", test_pantheraml_import),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        print()
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"📊 Test Summary:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        print("   ✅ torch_amp_custom_fwd AttributeError is fixed")
        print("   ✅ PantheraML can be imported successfully")
        print("   ✅ Device type handling works correctly")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed")
        print("   Please review the import fix")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
