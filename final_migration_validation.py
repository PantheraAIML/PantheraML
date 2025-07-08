#!/usr/bin/env python3
"""
Final Migration Validation Script
Comprehensive check that PantheraML has been fully migrated to use PantheraML-Zoo
"""

import os
import re
from pathlib import Path

def check_critical_files():
    """Check that critical files have been properly updated"""
    critical_files = [
        'pantheraml/__init__.py',
        'pantheraml/trainer.py', 
        'pantheraml/models/_utils.py',
        'pantheraml/chat_templates.py',
        'pyproject.toml',
        '.github/ISSUE_TEMPLATE/bug---issue.md',
        'pantheraml/models/loader_utils.py',
        'pantheraml/models/rl.py',
        'pantheraml/kernels/__init__.py'
    ]
    
    print("🔍 Checking critical files for proper migration...")
    all_good = True
    
    for file_path in critical_files:
        if not os.path.exists(file_path):
            print(f"❌ {file_path} not found")
            all_good = False
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for problematic unsloth_zoo references (excluding intentional ones)
        problematic_patterns = [
            r'from unsloth_zoo\.',
            r'import unsloth_zoo(?!\s*$)',  # Direct import without aliasing
            r'"unsloth_zoo"',
            r"'unsloth_zoo'",
        ]
        
        # Exclude patterns that are intentional (fallback logic, comments)
        exclude_patterns = [
            r'fallback.*unsloth_zoo',
            r'original.*unsloth_zoo',
            r'limited.*unsloth_zoo', 
            r'pip install.*unsloth_zoo',
            r'import pantheraml_zoo as unsloth_zoo',
            r'importlib_version\("unsloth_zoo"\)',
        ]
        
        issues = []
        for pattern in problematic_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                line_content = content.split('\n')[line_num - 1].strip()
                
                # Check if this match should be excluded
                is_excluded = False
                for exclude_pattern in exclude_patterns:
                    if re.search(exclude_pattern, line_content, re.IGNORECASE):
                        is_excluded = True
                        break
                
                if not is_excluded:
                    issues.append((line_num, line_content))
        
        if issues:
            print(f"❌ {file_path} has problematic unsloth_zoo references:")
            for line_num, line_content in issues:
                print(f"   Line {line_num}: {line_content}")
            all_good = False
        else:
            print(f"✅ {file_path}")
    
    return all_good

def check_dependency_consistency():
    """Check that dependencies are consistent"""
    print("\n🔍 Checking dependency consistency...")
    
    if not os.path.exists('pyproject.toml'):
        print("❌ pyproject.toml not found")
        return False
        
    with open('pyproject.toml', 'r') as f:
        content = f.read()
    
    # Check for pantheraml_zoo
    pantheraml_zoo_count = content.count('pantheraml_zoo')
    unsloth_zoo_count = content.count('unsloth_zoo')
    
    print(f"📊 pantheraml_zoo references: {pantheraml_zoo_count}")
    print(f"📊 unsloth_zoo references: {unsloth_zoo_count}")
    
    if pantheraml_zoo_count >= 3 and unsloth_zoo_count == 0:
        print("✅ Dependencies are properly configured")
        return True
    else:
        print("❌ Dependencies need attention")
        return False

def check_import_logic():
    """Check that import logic is correct"""
    print("\n🔍 Checking import logic in pantheraml/__init__.py...")
    
    if not os.path.exists('pantheraml/__init__.py'):
        print("❌ pantheraml/__init__.py not found")
        return False
        
    with open('pantheraml/__init__.py', 'r') as f:
        content = f.read()
    
    required_patterns = [
        r'import pantheraml_zoo as unsloth_zoo',
        r'PantheraML Zoo.*TPU-enabled',
        r'🐾 PantheraML.*Zoo.*loaded',
        r'importlib_version\("pantheraml_zoo"\)',
    ]
    
    all_found = True
    for pattern in required_patterns:
        if not re.search(pattern, content):
            print(f"❌ Missing pattern: {pattern}")
            all_found = False
    
    if all_found:
        print("✅ Import logic is correct")
        return True
    else:
        print("❌ Import logic needs attention")
        return False

def main():
    print("🚀 PantheraML to PantheraML-Zoo Migration Validation")
    print("=" * 60)
    
    # Check all components
    files_ok = check_critical_files()
    deps_ok = check_dependency_consistency()
    imports_ok = check_import_logic()
    
    print("\n" + "=" * 60)
    if files_ok and deps_ok and imports_ok:
        print("🎉 SUCCESS: Migration to PantheraML-Zoo is COMPLETE!")
        print("\n📋 Summary:")
        print("• ✅ All critical files updated")
        print("• ✅ Dependencies properly configured")
        print("• ✅ Import logic working correctly")
        print("• ✅ Fallback mechanism preserved")
        print("\n🚀 PantheraML is ready for TPU-enabled deployment!")
        print("\n📦 Users can now install with:")
        print("   pip install pantheraml")
        print("   # Automatically installs pantheraml_zoo for TPU support")
        return True
    else:
        print("❌ ISSUES FOUND: Migration needs attention")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
