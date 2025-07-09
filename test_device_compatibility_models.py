#!/usr/bin/env python3
"""
Test script to validate device compatibility across all PantheraML model files.
This ensures TPU, CUDA, XPU, and CPU support is properly implemented.
"""

import os
import sys
import importlib.util
from pathlib import Path

# Test device compatibility for different device types
DEVICE_TYPES_TO_TEST = ["cuda", "xpu", "tpu", "cpu"]

def test_device_compatibility():
    """Test device compatibility across all model files."""
    print("🔍 Testing device compatibility across all PantheraML models...")
    
    models_dir = Path(__file__).parent / "pantheraml" / "models"
    model_files = [f for f in models_dir.glob("*.py") if f.name not in ["__init__.py", "_utils.py"]]
    
    results = {}
    
    for device_type in DEVICE_TYPES_TO_TEST:
        print(f"\n📱 Testing device type: {device_type}")
        
        # Set device type environment
        os.environ["PANTHERAML_DEVICE_TYPE"] = device_type
        
        # Clear module cache to force re-import with new device type
        modules_to_clear = [m for m in sys.modules.keys() if m.startswith("pantheraml")]
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
        
        device_results = {}
        
        for model_file in model_files:
            model_name = model_file.stem
            print(f"  📄 Testing {model_name}.py...")
            
            try:
                # Import the model module
                spec = importlib.util.spec_from_file_location(f"pantheraml.models.{model_name}", model_file)
                module = importlib.util.module_from_spec(spec)
                
                # Check if the file has proper DEVICE_TYPE handling
                with open(model_file, 'r') as f:
                    content = f.read()
                
                issues = []
                
                # Check for hardcoded device references
                if '"cuda:0"' in content:
                    issues.append("Contains hardcoded 'cuda:0' device references")
                if '"cuda"' in content and 'device = "cuda"' in content:
                    issues.append("Contains hardcoded 'cuda' device references")
                
                # Check if DEVICE_TYPE is used for device allocation
                if 'torch.empty' in content and f'{device_type}:0' not in content and 'device = device' not in content:
                    # Allow dynamic device allocation or proper DEVICE_TYPE usage
                    if f'DEVICE_TYPE' not in content and device_type != "cpu":
                        issues.append("Missing DEVICE_TYPE usage for tensor allocation")
                
                # Try to import the module to check for import errors
                try:
                    spec.loader.exec_module(module)
                    import_success = True
                except Exception as e:
                    import_success = False
                    issues.append(f"Import failed: {str(e)}")
                
                device_results[model_name] = {
                    "issues": issues,
                    "import_success": import_success,
                    "status": "✅ PASS" if not issues and import_success else "❌ FAIL"
                }
                
                print(f"    {device_results[model_name]['status']}")
                if issues:
                    for issue in issues:
                        print(f"      ⚠️  {issue}")
                        
            except Exception as e:
                device_results[model_name] = {
                    "issues": [f"Test failed: {str(e)}"],
                    "import_success": False,
                    "status": "❌ ERROR"
                }
                print(f"    ❌ ERROR: {str(e)}")
        
        results[device_type] = device_results
    
    # Generate summary report
    print("\n" + "="*80)
    print("📊 DEVICE COMPATIBILITY TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for device_type, device_results in results.items():
        print(f"\n🔧 Device Type: {device_type}")
        print("-" * 40)
        
        passed = 0
        failed = 0
        
        for model_name, result in device_results.items():
            status = result['status']
            print(f"  {model_name:20} {status}")
            
            if "PASS" in status:
                passed += 1
            else:
                failed += 1
                all_passed = False
                
                # Show details for failed tests
                if result['issues']:
                    for issue in result['issues']:
                        print(f"    ⚠️  {issue}")
        
        print(f"\n  Summary: {passed} passed, {failed} failed")
    
    print(f"\n{'='*80}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! All model files are device-compatible.")
    else:
        print("❌ SOME TESTS FAILED! Please review the issues above.")
    print("="*80)
    
    return all_passed

def test_specific_device_patterns():
    """Test for specific device-related patterns that should be avoided."""
    print("\n🔍 Testing for specific device patterns...")
    
    models_dir = Path(__file__).parent / "pantheraml" / "models"
    model_files = [f for f in models_dir.glob("*.py") if f.name not in ["__init__.py"]]
    
    patterns_to_avoid = [
        ('"cuda:0"', "Hardcoded CUDA device"),
        ('"cuda"', "Hardcoded CUDA device (check context)"),
        ('device="cuda"', "Hardcoded CUDA device assignment"),
        ('device = "cuda"', "Hardcoded CUDA device assignment"),
    ]
    
    issues_found = False
    
    for model_file in model_files:
        with open(model_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        file_issues = []
        
        for pattern, description in patterns_to_avoid:
            for line_num, line in enumerate(lines, 1):
                if pattern in line:
                    # Skip comments and certain exceptions
                    if line.strip().startswith('#'):
                        continue
                    if 'device = "cuda"' in line and 'config' in line.lower():
                        continue  # Config-related lines might be OK
                        
                    file_issues.append((line_num, pattern, description, line.strip()))
        
        if file_issues:
            issues_found = True
            print(f"\n⚠️  Issues in {model_file.name}:")
            for line_num, pattern, description, line in file_issues:
                print(f"  Line {line_num}: {description}")
                print(f"    Pattern: {pattern}")
                print(f"    Code: {line}")
    
    if not issues_found:
        print("✅ No hardcoded device patterns found!")
    
    return not issues_found

if __name__ == "__main__":
    print("🚀 Starting device compatibility tests for PantheraML models...")
    
    # Test 1: General device compatibility
    compat_passed = test_device_compatibility()
    
    # Test 2: Specific pattern checking
    patterns_passed = test_specific_device_patterns()
    
    print(f"\n{'='*80}")
    print("🏁 FINAL RESULTS")
    print("="*80)
    print(f"Device Compatibility Tests: {'✅ PASSED' if compat_passed else '❌ FAILED'}")
    print(f"Device Pattern Tests:       {'✅ PASSED' if patterns_passed else '❌ FAILED'}")
    
    if compat_passed and patterns_passed:
        print("\n🎉 ALL DEVICE COMPATIBILITY TESTS PASSED!")
        print("All model files properly support TPU, CUDA, XPU, and CPU devices.")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please review and fix the issues identified above.")
        sys.exit(1)
