#!/usr/bin/env python3
"""
Test script to verify the PantheraMLVisionDataCollator fix
"""

def test_vision_data_collator_alias():
    """Test that PantheraMLVisionDataCollator alias is correctly defined"""
    try:
        # Read the trainer.py file to check if the alias exists
        with open('pantheraml/trainer.py', 'r') as f:
            content = f.read()
        
        if 'PantheraMLVisionDataCollator = UnslothVisionDataCollator' in content:
            print("✅ PantheraMLVisionDataCollator alias correctly defined in trainer.py")
            return True
        else:
            print("❌ PantheraMLVisionDataCollator alias not found in trainer.py")
            return False
            
    except Exception as e:
        print(f"❌ Error checking alias: {e}")
        return False

def test_all_exports():
    """Test that __all__ exports are correctly defined"""
    try:
        with open('pantheraml/trainer.py', 'r') as f:
            content = f.read()
        
        # Check that __all__ includes PantheraMLVisionDataCollator
        if '"PantheraMLVisionDataCollator"' in content:
            print("✅ PantheraMLVisionDataCollator correctly listed in __all__")
            return True
        else:
            print("❌ PantheraMLVisionDataCollator not found in __all__")
            return False
            
    except Exception as e:
        print(f"❌ Error checking __all__: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing PantheraMLVisionDataCollator fix...")
    print()
    
    results = []
    results.append(test_vision_data_collator_alias())
    results.append(test_all_exports())
    
    print()
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All tests passed! ({passed}/{total})")
        print("   PantheraMLVisionDataCollator should now be importable on GPU systems")
    else:
        print(f"❌ Some tests failed ({passed}/{total})")
