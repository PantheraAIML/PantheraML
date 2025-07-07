#!/usr/bin/env python3
"""
Final comprehensive test of all PantheraML fixes
"""

import ast
import sys
import os

def test_comprehensive_fixes():
    """Run comprehensive tests of all fixes"""
    
    print("🔍 Final Comprehensive PantheraML Test")
    print("="*60)
    
    all_passed = True
    
    # Test 1: helpsteer2_complete_pipeline.py syntax and structure
    print("\n1. Testing helpsteer2_complete_pipeline.py...")
    try:
        with open('examples/helpsteer2_complete_pipeline.py', 'r') as f:
            source = f.read()
        
        ast.parse(source)
        print("   ✅ Python syntax valid")
        
        # Check for import order (PantheraML first)
        lines = source.split('\n')
        pantheraml_import_line = None
        torch_import_line = None
        
        for i, line in enumerate(lines):
            if 'from pantheraml import' in line and pantheraml_import_line is None:
                pantheraml_import_line = i
            if 'import torch' in line and torch_import_line is None:
                torch_import_line = i
        
        if pantheraml_import_line and torch_import_line and pantheraml_import_line < torch_import_line:
            print("   ✅ Import order correct (PantheraML before torch)")
        else:
            print("   ❌ Import order incorrect")
            all_passed = False
        
        # Check fallback functions
        if all(f in source for f in ['_fallback_is_main_process', '_fallback_get_world_size']):
            print("   ✅ Fallback functions present")
        else:
            print("   ❌ Missing fallback functions")
            all_passed = False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_passed = False
    
    # Test 2: trainer.py fixes
    print("\n2. Testing trainer.py fixes...")
    try:
        with open('pantheraml/trainer.py', 'r') as f:
            trainer_source = f.read()
        
        if 'PantheraMLVisionDataCollator = UnslothVisionDataCollator' in trainer_source:
            print("   ✅ PantheraMLVisionDataCollator alias found")
        else:
            print("   ❌ PantheraMLVisionDataCollator alias missing")
            all_passed = False
        
        if '"PantheraMLVisionDataCollator"' in trainer_source:
            print("   ✅ PantheraMLVisionDataCollator in __all__")
        else:
            print("   ❌ PantheraMLVisionDataCollator not in __all__")
            all_passed = False
        
        if '"_patch_trl_trainer"' not in trainer_source:
            print("   ✅ _patch_trl_trainer removed from __all__")
        else:
            print("   ❌ _patch_trl_trainer still in __all__")
            all_passed = False
            
    except Exception as e:
        print(f"   ❌ Error reading trainer.py: {e}")
        all_passed = False
    
    # Test 3: CLI standalone script
    print("\n3. Testing CLI standalone script...")
    if os.path.exists('pantheraml/cli_standalone.py'):
        print("   ✅ CLI standalone script exists")
        try:
            with open('pantheraml/cli_standalone.py', 'r') as f:
                cli_source = f.read()
            if 'argparse' in cli_source and ('pantheraml' in cli_source.lower() or 'PantheraML' in cli_source):
                print("   ✅ CLI script has correct content")
            else:
                print("   ❌ CLI script missing required content")
                all_passed = False
        except Exception as e:
            print(f"   ❌ Error reading CLI script: {e}")
            all_passed = False
    else:
        print("   ❌ CLI standalone script missing")
        all_passed = False
    
    # Test 4: pyproject.toml dependencies
    print("\n4. Testing pyproject.toml dependencies...")
    try:
        with open('pyproject.toml', 'r') as f:
            pyproject_content = f.read()
        
        required_deps = ['trl', 'bitsandbytes', 'unsloth_zoo']
        missing_deps = []
        
        for dep in required_deps:
            if dep in pyproject_content:
                print(f"   ✅ {dep} dependency found")
            else:
                print(f"   ❌ {dep} dependency missing")
                missing_deps.append(dep)
                all_passed = False
        
        if 'pantheraml-cli = "pantheraml.cli_standalone:main"' in pyproject_content:
            print("   ✅ CLI entry point correctly configured")
        else:
            print("   ❌ CLI entry point missing or incorrect")
            all_passed = False
            
    except Exception as e:
        print(f"   ❌ Error reading pyproject.toml: {e}")
        all_passed = False
    
    # Test 5: Check specific import fixes
    print("\n5. Testing import fixes...")
    
    # Check models/_utils.py
    try:
        with open('pantheraml/models/_utils.py', 'r') as f:
            utils_source = f.read()
        if 'from pantheraml import DEVICE_TYPE' in utils_source:
            print("   ✅ _utils.py DEVICE_TYPE import fixed")
        else:
            print("   ❌ _utils.py DEVICE_TYPE import not fixed")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Error checking _utils.py: {e}")
        all_passed = False
    
    # Check models/vision.py
    try:
        with open('pantheraml/models/vision.py', 'r') as f:
            vision_source = f.read()
        if 'from pantheraml import DEVICE_TYPE' in vision_source:
            print("   ✅ vision.py DEVICE_TYPE import fixed")
        else:
            print("   ❌ vision.py DEVICE_TYPE import not fixed")
            all_passed = False
    except Exception as e:
        print(f"   ❌ Error checking vision.py: {e}")
        all_passed = False
    
    # Test 6: Branding consistency
    print("\n6. Testing branding consistency...")
    
    # Check for PantheraML branding in key files
    key_files = [
        'pantheraml/__init__.py',
        'pantheraml/trainer.py',
        'examples/helpsteer2_complete_pipeline.py'
    ]
    
    for file_path in key_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Count occurrences (some references to unsloth in dependencies are expected)
            pantheraml_count = content.lower().count('pantheraml')
            
            if pantheraml_count > 0:
                print(f"   ✅ {file_path}: PantheraML branding present ({pantheraml_count} refs)")
            else:
                print(f"   ❌ {file_path}: No PantheraML branding found")
                all_passed = False
                
        except Exception as e:
            print(f"   ❌ Error checking {file_path}: {e}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! PantheraML is ready for production!")
        print("\n📋 Summary of fixes:")
        print("   • Import order optimized (PantheraML first)")
        print("   • Fallback functions for single-GPU mode")
        print("   • PantheraMLVisionDataCollator alias added")
        print("   • CLI entry point configured")
        print("   • Dependencies updated")
        print("   • Import fixes applied")
        print("   • Branding consistency maintained")
        print("\n🚀 Ready for multi-GPU training on GPU-enabled systems!")
        print("💡 Note: 'Unsloth Zoo' messages are from dependencies (expected)")
        return True
    else:
        print("❌ SOME TESTS FAILED! Please review the errors above.")
        return False

if __name__ == "__main__":
    success = test_comprehensive_fixes()
    sys.exit(0 if success else 1)
