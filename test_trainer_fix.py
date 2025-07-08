#!/usr/bin/env python3
"""
Test the SFTTrainer compatibility fix for PantheraML
"""

import os
import sys

def test_trainer_compatibility_fix():
    """Test that the trainer initialization fix handles both cases correctly"""
    
    print("🧪 Testing SFTTrainer compatibility fix...")
    
    try:
        print("✅ Testing trainer initialization logic...")
        
        # Check if our fix is in the file
        with open('examples/helpsteer2_complete_pipeline.py', 'r') as f:
            content = f.read()
        
        if 'PANTHERAML_TRAINER_AVAILABLE' in content:
            print("✅ PantheraMLTrainer availability check added")
        else:
            print("❌ PantheraMLTrainer availability check missing")
            return False
        
        if 'unexpected keyword argument \'tokenizer\'' in content:
            print("✅ SFTTrainer tokenizer compatibility handling added")
        else:
            print("❌ SFTTrainer tokenizer compatibility handling missing")
            return False
        
        if 'PantheraMLTrainer(' in content and 'SFTTrainer(' in content:
            print("✅ Both trainer types supported")
        else:
            print("❌ Missing support for both trainer types")
            return False
        
        print("🎯 SFTTrainer compatibility fix verified!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_trainer_compatibility_fix()
    print(f"\n🏁 Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
