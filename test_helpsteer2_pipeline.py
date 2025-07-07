#!/usr/bin/env python3
"""
Test script to validate the HelpSteer2 complete pipeline
"""

import sys
import os
import subprocess
import importlib.util
import ast

def test_script_syntax():
    """Test that the script has valid Python syntax"""
    script_path = "examples/helpsteer2_complete_pipeline.py"
    
    print("🔍 Testing script syntax...")
    
    # Check if file exists
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Test syntax compilation
    try:
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        compile(script_content, script_path, 'exec')
        print("✅ Script syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in script: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading script: {e}")
        return False

def test_imports():
    """Test that required imports are available"""
    print("\n🔍 Testing required imports...")
    
    required_modules = [
        "torch",
        "transformers", 
        "datasets",
        "trl"
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} (not installed)")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️ Missing modules: {missing_modules}")
        print("Install with: pip install torch transformers datasets trl")
        return False
    
    print("✅ All required modules are available")
    return True

def test_pantheraml_imports():
    """Test PantheraML-specific imports"""
    print("\n🔍 Testing PantheraML imports...")
    
    try:
        # Test that the files exist and have correct syntax
        import ast
        
        files_to_check = [
            "pantheraml/__init__.py",
            "pantheraml/chat_templates.py", 
            "pantheraml/trainer.py"
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    ast.parse(content)
                    print(f"✅ {file_path} syntax valid")
                except SyntaxError as e:
                    print(f"❌ {file_path} syntax error: {e}")
                    return False
            else:
                print(f"❌ {file_path} not found")
                return False
        
        print("✅ PantheraML module structure is valid")
        print("ℹ️ Note: Full import test skipped (requires GPU/TPU)")
        return True
        
    except Exception as e:
        print(f"❌ Error checking PantheraML structure: {e}")
        return False

def test_help_output():
    """Test that the script shows help correctly"""
    print("\n🔍 Testing script help output...")
    
    try:
        result = subprocess.run([
            sys.executable, "examples/helpsteer2_complete_pipeline.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Script help works correctly")
            print("Help output preview:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"❌ Script help failed with return code: {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Script help timed out")
        return False
    except Exception as e:
        print(f"❌ Error running script help: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing HelpSteer2 Complete Pipeline Script")
    print("=" * 50)
    
    tests = [
        ("Script Syntax", test_script_syntax),
        ("Required Imports", test_imports),
        ("PantheraML Imports", test_pantheraml_imports),
        ("Help Output", test_help_output)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! The HelpSteer2 pipeline script is ready to use.")
        print("\nTo run the full pipeline:")
        print("  python examples/helpsteer2_complete_pipeline.py --max_steps 10 --max_samples 100")
    else:
        print(f"\n⚠️ {len(tests) - passed} test(s) failed. Please check the issues above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
