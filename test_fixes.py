#!/usr/bin/env python3
"""
Test script to verify the key import fixes are working
"""

import os
os.environ["PANTHERAML_DEV_MODE"] = "1"

def test_basic_imports():
    """Test basic imports without full model loading"""
    try:
        # Test that the device detection works in dev mode
        from pantheraml import DEVICE_TYPE
        print(f"✅ Device detection working: {DEVICE_TYPE}")
        return True
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False

def test_trainer_import():
    """Test trainer import specifically"""
    try:
        # This should not have the _patch_trl_trainer error anymore
        from pantheraml.trainer import PantheraMLTrainer
        print("✅ Trainer import successful")
        return True
    except AttributeError as e:
        if "_patch_trl_trainer" in str(e):
            print(f"❌ _patch_trl_trainer error still exists: {e}")
            return False
        else:
            print(f"❌ Other AttributeError: {e}")
            return False
    except Exception as e:
        print(f"❌ Other trainer import error: {e}")
        return False

def test_cli_help():
    """Test CLI help functionality"""
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, 
            "pantheraml/cli_standalone.py", 
            "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "PantheraML CLI" in result.stdout:
            print("✅ CLI help working")
            return True
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ CLI help test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing PantheraML fixes...")
    print()
    
    results = []
    results.append(test_basic_imports())
    results.append(test_trainer_import())
    results.append(test_cli_help())
    
    print()
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print()
        print("✅ Key fixes working:")
        print("   • Device detection with development mode")
        print("   • Trainer module imports correctly") 
        print("   • CLI help functionality works")
        print("   • No more _patch_trl_trainer errors")
        print()
        print("🚀 Ready for deployment to Kaggle/production!")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        print("   More fixes may be needed")
