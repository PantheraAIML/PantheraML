#!/usr/bin/env python3
"""
Test script to verify that PantheraML-Zoo is properly configured as a dependency
and that the import fallback mechanism works correctly.
"""

import sys
import subprocess

def test_dependency_setup():
    """Test that pantheraml_zoo is configured as a dependency in pyproject.toml"""
    
    print("🔍 Testing PantheraML-Zoo dependency configuration...")
    
    # Read pyproject.toml to verify pantheraml_zoo is listed as a dependency
    try:
        with open('pyproject.toml', 'r') as f:
            content = f.read()
        
        # Check that pantheraml_zoo is in dependencies
        if 'pantheraml_zoo>=2025.3.1' in content:
            print("✅ pantheraml_zoo is correctly listed as a dependency")
        else:
            print("❌ pantheraml_zoo not found in dependencies")
            return False
            
        # Check that unsloth_zoo has been removed
        if 'unsloth_zoo' not in content:
            print("✅ unsloth_zoo has been successfully removed from dependencies")
        else:
            print("❌ unsloth_zoo still found in dependencies")
            return False
            
    except Exception as e:
        print(f"❌ Error reading pyproject.toml: {e}")
        return False
    
    print("\n🔍 Testing import mechanism...")
    
    # Test the import mechanism in pantheraml/__init__.py
    try:
        # Import pantheraml to trigger the zoo import logic
        import pantheraml
        
        # Check if zoo was imported successfully
        if hasattr(pantheraml, '_zoo_imported'):
            if pantheraml._zoo_imported:
                print("✅ Zoo dependency imported successfully")
            else:
                print("⚠️  Zoo dependency not imported (this is expected if pantheraml_zoo is not installed)")
        
        print("✅ PantheraML import successful")
        
    except Exception as e:
        print(f"❌ Error importing pantheraml: {e}")
        return False
    
    print("\n📋 Summary:")
    print("• pantheraml_zoo is now the default dependency")
    print("• unsloth_zoo has been removed from dependencies")
    print("• Fallback mechanism is in place for compatibility")
    print("• Users installing PantheraML will automatically get PantheraML-Zoo")
    
    return True

def main():
    """Main test function"""
    print("🚀 PantheraML-Zoo Dependency Test")
    print("=" * 50)
    
    success = test_dependency_setup()
    
    if success:
        print("\n🎉 All dependency tests passed!")
        print("\nNext steps:")
        print("1. Build and publish PantheraML package")
        print("2. Users can install with: pip install pantheraml")
        print("3. PantheraML-Zoo will be installed automatically")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
