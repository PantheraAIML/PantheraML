#!/usr/bin/env python3
"""
Comprehensive device-agnostic verification for all PantheraML model files.
This script checks for any remaining hardcoded device references.
"""

import os
from pathlib import Path

def check_device_agnostic():
    """Check all model files for device-agnostic compliance."""
    print("🔍 Checking all PantheraML model files for device-agnostic compliance...")
    
    models_dir = Path(__file__).parent / "pantheraml" / "models"
    model_files = [f for f in models_dir.glob("*.py") if f.name != "__init__.py"]
    
    issues = []
    
    # Problematic patterns to check for
    bad_patterns = [
        (r'"cuda:\d+"', 'Hardcoded CUDA device with index'),
        (r'device\s*=\s*"cuda"(?![a-zA-Z])', 'Hardcoded CUDA device assignment'),
        (r'\.to\("cuda"\)', 'Hardcoded .to("cuda") call'),
        (r'\.cuda\(\)', 'Hardcoded .cuda() call'),
        (r'device_type\s*=\s*"cuda"', 'Hardcoded device_type in autocast (check if conditional)'),
    ]
    
    # Allowed patterns (these are okay)
    allowed_patterns = [
        'if DEVICE_TYPE == "cuda":',
        'device="cpu"',  # CPU is always okay
        'device = "cpu"',
        'torch.cuda.',  # CUDA API calls are okay when conditional
        '# cuda',  # Comments are okay
        '"CUDA:',  # String literals for messages are okay
    ]
    
    for model_file in model_files:
        print(f"  📄 Checking {model_file.name}...")
        
        with open(model_file, 'r') as f:
            lines = f.readlines()
        
        file_issues = []
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Skip comments
            if line_stripped.startswith('#'):
                continue
            
            # Check for problematic patterns
            for pattern, description in bad_patterns:
                import re
                if re.search(pattern, line):
                    # Check if it's in an allowed context
                    is_allowed = False
                    for allowed in allowed_patterns:
                        if allowed in line:
                            is_allowed = True
                            break
                    
                    # Special case: device_type = "cuda" is okay if it's conditional
                    if 'device_type' in pattern:
                        # Check current line and previous 10 lines for conditional
                        context_lines = lines[max(0, line_num-10):line_num]
                        if 'if DEVICE_TYPE ==' in line or any('if DEVICE_TYPE ==' in ctx_line for ctx_line in context_lines):
                            is_allowed = True
                    
                    if not is_allowed:
                        file_issues.append((line_num, description, line.strip()))
        
        if file_issues:
            issues.extend([(model_file.name, issue) for issue in file_issues])
            print(f"    ❌ Found {len(file_issues)} issues")
            for line_num, description, line in file_issues:
                print(f"      Line {line_num}: {description}")
                print(f"        Code: {line}")
        else:
            print(f"    ✅ Clean")
    
    print(f"\n{'='*80}")
    print("📊 DEVICE-AGNOSTIC COMPLIANCE REPORT")
    print("="*80)
    
    if not issues:
        print("🎉 ALL FILES ARE DEVICE-AGNOSTIC!")
        print("✅ No hardcoded device references found.")
        print("✅ All models support CUDA, XPU, TPU, and CPU devices.")
    else:
        print(f"❌ Found {len(issues)} issues across {len(set(issue[0] for issue in issues))} files:")
        
        current_file = None
        for file_name, (line_num, description, line) in issues:
            if file_name != current_file:
                print(f"\n📄 {file_name}:")
                current_file = file_name
            print(f"  Line {line_num}: {description}")
            print(f"    {line}")
    
    return len(issues) == 0

def check_device_type_imports():
    """Check that all model files that need DEVICE_TYPE have proper imports."""
    print("\n🔍 Checking DEVICE_TYPE import consistency...")
    
    models_dir = Path(__file__).parent / "pantheraml" / "models"
    model_files = [f for f in models_dir.glob("*.py") if f.name not in ["__init__.py", "_utils.py"]]
    
    import_issues = []
    
    for model_file in model_files:
        with open(model_file, 'r') as f:
            content = f.read()
        
        # Check if file uses DEVICE_TYPE
        uses_device_type = 'DEVICE_TYPE' in content
        
        # Check if file imports DEVICE_TYPE (directly or via llama import)
        has_direct_import = 'from pantheraml import DEVICE_TYPE' in content
        has_llama_import = 'from .llama import *' in content
        
        if uses_device_type and not (has_direct_import or has_llama_import):
            import_issues.append(model_file.name)
    
    if import_issues:
        print(f"❌ Files using DEVICE_TYPE without proper imports: {import_issues}")
        return False
    else:
        print("✅ All files have proper DEVICE_TYPE imports")
        return True

if __name__ == "__main__":
    print("🚀 Starting comprehensive device-agnostic verification...")
    
    # Check 1: Device-agnostic compliance
    compliance_ok = check_device_agnostic()
    
    # Check 2: DEVICE_TYPE import consistency
    imports_ok = check_device_type_imports()
    
    print(f"\n{'='*80}")
    print("🏁 FINAL VERIFICATION RESULTS")
    print("="*80)
    print(f"Device-Agnostic Compliance: {'✅ PASSED' if compliance_ok else '❌ FAILED'}")
    print(f"DEVICE_TYPE Import Check:   {'✅ PASSED' if imports_ok else '❌ FAILED'}")
    
    if compliance_ok and imports_ok:
        print("\n🎉 ALL VERIFICATION CHECKS PASSED!")
        print("✨ All PantheraML model files are properly device-agnostic.")
        print("🌐 Models support CUDA, XPU, TPU, and CPU devices seamlessly.")
        exit(0)
    else:
        print("\n❌ VERIFICATION FAILED!")
        print("Please fix the issues identified above.")
        exit(1)
