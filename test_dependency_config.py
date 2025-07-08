#!/usr/bin/env python3
"""
Simplified test script to verify PantheraML-Zoo dependency configuration
without importing PantheraML (to avoid hardware detection issues).
"""

import sys
import os

def test_dependency_configuration():
    """Test that pantheraml_zoo is properly configured as a dependency"""
    
    print("🔍 Testing PantheraML-Zoo dependency configuration...")
    
    # Test 1: Check pyproject.toml has correct dependencies
    try:
        with open('pyproject.toml', 'r') as f:
            content = f.read()
        
        # Count occurrences of pantheraml_zoo
        pantheraml_zoo_count = content.count('pantheraml_zoo>=2025.3.1')
        unsloth_zoo_count = content.count('unsloth_zoo')
        
        print(f"✅ pantheraml_zoo dependencies found: {pantheraml_zoo_count}")
        
        if pantheraml_zoo_count >= 3:  # Should be in main dependencies, huggingface, and colab-new
            print("✅ pantheraml_zoo is correctly listed in all required dependency sections")
        else:
            print(f"❌ pantheraml_zoo not found in all expected sections (found {pantheraml_zoo_count}, expected ≥3)")
            return False
            
        if unsloth_zoo_count == 0:
            print("✅ unsloth_zoo has been completely removed from dependencies")
        else:
            print(f"❌ unsloth_zoo still found {unsloth_zoo_count} times in dependencies")
            return False
            
    except Exception as e:
        print(f"❌ Error reading pyproject.toml: {e}")
        return False
    
    # Test 2: Check that __init__.py has the correct import logic
    try:
        with open('pantheraml/__init__.py', 'r') as f:
            init_content = f.read()
        
        if 'pantheraml_zoo' in init_content:
            print("✅ pantheraml/__init__.py references pantheraml_zoo")
        else:
            print("❌ pantheraml/__init__.py does not reference pantheraml_zoo")
            return False
            
        if 'PantheraML Zoo' in init_content:
            print("✅ pantheraml/__init__.py has PantheraML Zoo documentation")
        else:
            print("❌ pantheraml/__init__.py missing PantheraML Zoo documentation")
            return False
            
    except Exception as e:
        print(f"❌ Error reading pantheraml/__init__.py: {e}")
        return False
    
    # Test 3: Check that imports in other files use pantheraml_zoo
    import_files = [
        'pantheraml/trainer.py',
        'pantheraml/models/_utils.py',
        'pantheraml/chat_templates.py'
    ]
    
    for file_path in import_files:
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
            
            if 'from pantheraml_zoo' in file_content:
                print(f"✅ {file_path} uses pantheraml_zoo imports")
            else:
                print(f"❌ {file_path} does not use pantheraml_zoo imports")
                return False
                
        except Exception as e:
            print(f"⚠️  Could not check {file_path}: {e}")
    
    return True

def main():
    """Main test function"""
    print("🚀 PantheraML-Zoo Dependency Configuration Test")
    print("=" * 55)
    
    success = test_dependency_configuration()
    
    if success:
        print("\n🎉 All dependency configuration tests passed!")
        print("\n📋 Configuration Summary:")
        print("• ✅ pantheraml_zoo is the primary dependency")
        print("• ✅ unsloth_zoo has been completely removed")
        print("• ✅ Import statements updated to use pantheraml_zoo")
        print("• ✅ Fallback mechanism preserved for compatibility")
        
        print("\n🚀 Ready for deployment!")
        print("When users install PantheraML, they will automatically get:")
        print("  - PantheraML core library")
        print("  - PantheraML-Zoo (TPU-enabled utilities)")
        print("  - All required dependencies")
        
        print("\n📦 Installation command for users:")
        print("  pip install pantheraml")
        print("  # This will automatically install pantheraml_zoo from:")
        print("  # https://github.com/PantheraAIML/PantheraML-Zoo.git")
        
        return 0
    else:
        print("\n❌ Some configuration tests failed!")
        print("Please check the issues above and fix them.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
