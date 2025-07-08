#!/usr/bin/env python3
"""
Test the gradient checkpointing fix for PantheraML
"""

import os
import sys

def test_gradient_checkpointing_fix():
    """Test that the gradient checkpointing parameter is correctly set"""
    
    print("🧪 Testing gradient checkpointing fix...")
    
    # Simulate Kaggle environment for testing
    if not os.path.exists('/kaggle'):
        print("📁 Creating mock Kaggle directory for testing...")
        try:
            os.makedirs('/tmp/mock_kaggle', exist_ok=True)
            # Temporarily set an environment variable to simulate Kaggle
            os.environ['KAGGLE_TEST_MODE'] = '1'
        except:
            pass
    
    # Test the function call without actually running it (mock test)
    try:
        print("✅ Testing gradient checkpointing parameter...")
        
        # Check if our fix is in the file
        with open('examples/helpsteer2_complete_pipeline.py', 'r') as f:
            content = f.read()
        
        if 'use_gradient_checkpointing="unsloth"' in content:
            print("✅ Gradient checkpointing parameter fixed: 'unsloth' is used")
        else:
            print("❌ Gradient checkpointing parameter not fixed")
            return False
        
        if 'use_gradient_checkpointing="pantheraml"' in content:
            print("❌ Still contains old 'pantheraml' parameter")
            return False
        else:
            print("✅ Old 'pantheraml' parameter removed")
        
        print("🎯 Gradient checkpointing fix verified!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gradient_checkpointing_fix()
    print(f"\n🏁 Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
