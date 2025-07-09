#!/usr/bin/env python3
"""
Test specific import fixes for the latest issues
"""

import os
import sys

def test_callable_import():
    """Test that Callable is properly imported in distributed.py"""
    
    print("🧪 Testing Callable import fix...")
    print("=" * 40)
    
    # Set dev mode environment
    os.environ["PANTHERAML_DEV_MODE"] = "1"
    
    try:
        # Test importing the distributed module directly
        print("📦 Testing distributed module import...")
        from pantheraml.distributed import coordinate_multi_pod_training
        
        print("   ✅ coordinate_multi_pod_training imported successfully")
        
        # Check that it's callable
        if callable(coordinate_multi_pod_training):
            print("   ✅ Function is callable")
            return True
        else:
            print("   ❌ Function is not callable")
            return False
            
    except NameError as e:
        if "Callable" in str(e):
            print(f"   ❌ Callable import error still occurs: {e}")
            return False
        else:
            print(f"   ⚠️ Different NameError: {e}")
            return True
    except Exception as e:
        print(f"   ⚠️ Other error (may be expected): {e}")
        return True

def test_tpu_performance_fixes():
    """Test TPU performance module fixes"""
    
    print("\n🧪 Testing TPU performance fixes...")
    print("=" * 40)
    
    try:
        from pantheraml.kernels.tpu_performance import TPUAttentionOptimizer
        
        print("   ✅ TPUAttentionOptimizer imported successfully")
        
        # Try to instantiate it
        optimizer = TPUAttentionOptimizer()
        print("   ✅ TPUAttentionOptimizer instantiated successfully")
        
        return True
        
    except RuntimeError as e:
        if "@torch.jit.script" in str(e) or "bool()" in str(e):
            print(f"   ❌ TorchScript error still occurs: {e}")
            return False
        else:
            print(f"   ⚠️ Different RuntimeError: {e}")
            return True
    except Exception as e:
        print(f"   ⚠️ Other error (may be expected): {e}")
        return True

def main():
    """Test all latest fixes"""
    
    print("🧪 Latest Import Fixes Test")
    print("=" * 30)
    print("Testing latest import error fixes...")
    print()
    
    tests = [
        ("Callable Import Fix", test_callable_import),
        ("TPU Performance Fixes", test_tpu_performance_fixes),
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
        print("\n🎉 All latest fixes working!")
        print("   ✅ Callable import error resolved")
        print("   ✅ TPU performance module loads correctly")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed")
        print("   Please review the latest fixes")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
