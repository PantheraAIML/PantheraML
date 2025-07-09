#!/usr/bin/env python3
"""
Simple test to check for hardcoded device references in model files.
"""

import os
from pathlib import Path

def check_hardcoded_devices():
    """Check for hardcoded device references across all model files."""
    print("🔍 Checking for hardcoded device references...")
    
    models_dir = Path(__file__).parent / "pantheraml" / "models"
    model_files = [f for f in models_dir.glob("*.py") if f.name not in ["__init__.py"]]
    
    patterns_to_avoid = [
        '"cuda:0"',
        'device="cuda"',
        'device = "cuda"',
    ]
    
    # Allowed exceptions
    allowed_exceptions = [
        'device_type = "cuda"',  # torch.autocast usage
        'if DEVICE_TYPE == "cuda"',  # conditional checks
        'elif DEVICE_TYPE == "cuda"',  # conditional checks
        'torch.device("cuda"',  # explicit device creation
    ]
    
    issues_found = False
    total_issues = 0
    
    for model_file in model_files:
        with open(model_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        file_issues = []
        
        for pattern in patterns_to_avoid:
            for line_num, line in enumerate(lines, 1):
                if pattern in line:
                    # Skip comments
                    if line.strip().startswith('#'):
                        continue
                    
                    # Skip allowed exceptions
                    is_exception = any(exception in line for exception in allowed_exceptions)
                    if is_exception:
                        continue
                        
                    file_issues.append((line_num, pattern, line.strip()))
        
        if file_issues:
            issues_found = True
            print(f"\n❌ Issues in {model_file.name}:")
            for line_num, pattern, line in file_issues:
                print(f"  Line {line_num}: {pattern}")
                print(f"    Code: {line}")
                total_issues += 1
    
    print(f"\n{'='*60}")
    if not issues_found:
        print("✅ No hardcoded device references found!")
        print("All model files properly use DEVICE_TYPE for device allocation.")
    else:
        print(f"❌ Found {total_issues} hardcoded device reference(s)!")
        print("Please replace them with f'{DEVICE_TYPE}:0' or dynamic device allocation.")
    print("="*60)
    
    return not issues_found

if __name__ == "__main__":
    success = check_hardcoded_devices()
    exit(0 if success else 1)
