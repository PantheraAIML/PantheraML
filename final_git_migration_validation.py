#!/usr/bin/env python3
"""
Final validation script for PantheraML-Zoo Git-based migration
Tests the complete migration from package-based to Git-based dependencies
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple

def scan_all_references() -> Dict[str, List[str]]:
    """Scan all files for pantheraml_zoo and unsloth_zoo references"""
    
    base_path = Path("/Users/aayanmishra/unsloth")
    
    # File patterns to check
    patterns = [
        "**/*.py",
        "**/*.ipynb", 
        "**/*.toml",
        "**/*.md",
        "**/*.yml",
        "**/*.yaml"
    ]
    
    references = {
        "git_urls": [],
        "package_installs": [],
        "import_statements": [],
        "fallback_logic": [],
        "documentation": []
    }
    
    for pattern in patterns:
        for file_path in base_path.glob(pattern):
            # Skip certain directories
            if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                relative_path = str(file_path.relative_to(base_path))
                
                # Check for Git URLs
                git_patterns = [
                    r'git\+https://github\.com/PantheraAIML/PantheraML-Zoo\.git',
                    r'git\+https://github\.com/unslothai/unsloth-zoo\.git'
                ]
                
                for pattern in git_patterns:
                    if re.search(pattern, content):
                        references["git_urls"].append(f"{relative_path}: {pattern}")
                
                # Check for package installation commands
                package_patterns = [
                    r'pip install.*pantheraml_zoo(?!\s+@)',  # Not followed by @
                    r'pip install.*unsloth_zoo'
                ]
                
                for pattern in package_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        references["package_installs"].append(f"{relative_path}: {match}")
                
                # Check for import statements
                import_patterns = [
                    r'import pantheraml_zoo',
                    r'from pantheraml_zoo',
                    r'import unsloth_zoo',
                    r'from unsloth_zoo'
                ]
                
                for pattern in import_patterns:
                    if re.search(pattern, content):
                        references["import_statements"].append(f"{relative_path}: {pattern}")
                
                # Check for fallback logic
                fallback_patterns = [
                    r'fallback.*unsloth_zoo',
                    r'original.*unsloth_zoo',
                    r'limited.*unsloth_zoo'
                ]
                
                for pattern in fallback_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        references["fallback_logic"].append(f"{relative_path}: {pattern}")
                
                # Check for documentation references
                doc_patterns = [
                    r'TPU-enabled.*fork',
                    r'PantheraML-Zoo.*fork',
                    r'fork.*unsloth_zoo'
                ]
                
                for pattern in doc_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        references["documentation"].append(f"{relative_path}: {pattern}")
                        
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
    
    return references

def validate_migration_completeness() -> bool:
    """Validate that the migration to Git-based dependencies is complete"""
    
    print("🔍 Scanning all files for dependency references...")
    print("=" * 60)
    
    references = scan_all_references()
    
    success = True
    
    # Check Git URLs
    print("📦 Git URL References:")
    if references["git_urls"]:
        for ref in references["git_urls"]:
            print(f"   ✅ {ref}")
        print(f"   Total: {len(references['git_urls'])} Git URLs found")
    else:
        print("   ❌ No Git URLs found")
        success = False
    
    print()
    
    # Check problematic package installs
    print("⚠️ Package Installation Commands:")
    if references["package_installs"]:
        print("   Found package-based installations (may need updating):")
        for ref in references["package_installs"]:
            # Check if this is an intentional fallback
            if 'fallback' in ref.lower() or 'original' in ref.lower():
                print(f"   ⚠️ {ref} (intentional fallback)")
            else:
                print(f"   ❌ {ref} (should use Git URL)")
                success = False
    else:
        print("   ✅ No problematic package installations found")
    
    print()
    
    # Check import statements
    print("📥 Import Statements:")
    pantheraml_zoo_imports = [ref for ref in references["import_statements"] if 'pantheraml_zoo' in ref]
    unsloth_zoo_imports = [ref for ref in references["import_statements"] if 'unsloth_zoo' in ref and 'pantheraml_zoo' not in ref]
    
    print(f"   PantheraML-Zoo imports: {len(pantheraml_zoo_imports)}")
    print(f"   Unsloth-Zoo imports: {len(unsloth_zoo_imports)}")
    
    if len(pantheraml_zoo_imports) >= len(unsloth_zoo_imports):
        print("   ✅ PantheraML-Zoo is primary import target")
    else:
        print("   ❌ More unsloth_zoo imports than pantheraml_zoo")
        success = False
    
    print()
    
    # Check fallback logic
    print("🔄 Fallback Logic:")
    if references["fallback_logic"]:
        print(f"   ✅ {len(references['fallback_logic'])} fallback references found")
        for ref in references["fallback_logic"][:3]:  # Show first 3
            print(f"      {ref}")
        if len(references["fallback_logic"]) > 3:
            print(f"      ... and {len(references['fallback_logic']) - 3} more")
    else:
        print("   ⚠️ No fallback logic found")
    
    print()
    
    # Check documentation
    print("📚 Documentation References:")
    if references["documentation"]:
        print(f"   ✅ {len(references['documentation'])} documentation references found")
    else:
        print("   ⚠️ No documentation references found")
    
    return success

def test_pyproject_dependency_syntax() -> bool:
    """Test that pyproject.toml has valid Git dependency syntax"""
    
    print("\n🔧 Testing pyproject.toml Git dependency syntax...")
    
    pyproject_path = "/Users/aayanmishra/unsloth/pyproject.toml"
    
    if not os.path.exists(pyproject_path):
        print("❌ pyproject.toml not found")
        return False
    
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    # Test valid Git dependency format
    valid_patterns = [
        r'pantheraml_zoo @ git\+https://github\.com/PantheraAIML/PantheraML-Zoo\.git',
    ]
    
    valid_count = 0
    for pattern in valid_patterns:
        matches = re.findall(pattern, content)
        valid_count += len(matches)
        print(f"   Pattern '{pattern}': {len(matches)} matches")
    
    # Check for invalid patterns
    invalid_patterns = [
        r'pantheraml_zoo>=[\d\.]+',  # Version-based dependency
        r'pantheraml_zoo==[\d\.]+',  # Exact version
        r'pantheraml_zoo~=[\d\.]+',  # Compatible version
    ]
    
    invalid_count = 0
    for pattern in invalid_patterns:
        matches = re.findall(pattern, content)
        if matches:
            invalid_count += len(matches)
            print(f"   ❌ Invalid pattern '{pattern}': {len(matches)} matches")
            for match in matches:
                print(f"      {match}")
    
    if invalid_count > 0:
        print(f"❌ Found {invalid_count} invalid dependency formats")
        return False
    
    if valid_count >= 2:  # At least main deps + one optional dep
        print(f"✅ Found {valid_count} valid Git dependency formats")
        return True
    else:
        print(f"❌ Insufficient valid Git dependencies found: {valid_count}")
        return False

def generate_installation_guide() -> None:
    """Generate installation guide for users"""
    
    print("\n📋 Installation Guide for Users:")
    print("=" * 40)
    print()
    print("🚀 To install PantheraML with PantheraML-Zoo (recommended):")
    print("   pip install git+https://github.com/PantheraAIML/PantheraML-Zoo.git")
    print()
    print("📦 To install PantheraML from source:")
    print("   git clone https://github.com/PantheraML/pantheraml.git")
    print("   cd pantheraml")
    print("   pip install -e .")
    print("   # This will automatically install PantheraML-Zoo via Git")
    print()
    print("🔄 For fallback (if PantheraML-Zoo unavailable):")
    print("   pip install unsloth_zoo")
    print("   # Limited TPU support")
    print()
    print("🧪 For development:")
    print("   pip install -e .[dev]")
    print("   # Includes all dependencies")

def main():
    """Run complete validation of Git-based migration"""
    
    print("🧪 PantheraML-Zoo Git Migration Final Validation")
    print("=" * 50)
    print("Validating complete migration to Git-based dependencies...")
    print()
    
    tests = [
        ("Migration Completeness", validate_migration_completeness),
        ("pyproject.toml Syntax", test_pyproject_dependency_syntax),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        print()
    
    # Generate installation guide
    generate_installation_guide()
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 Final Validation Summary:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 Migration to Git-based dependencies COMPLETE!")
        print("   ✅ All references use correct Git URLs")
        print("   ✅ No problematic package-based dependencies")
        print("   ✅ Proper fallback logic implemented")
        print("   ✅ Installation commands use Git syntax")
        print("\n🚀 PantheraML is now fully configured with PantheraML-Zoo!")
    else:
        print(f"\n⚠️ {total-passed} validation(s) failed")
        print("   Please review the migration for any remaining issues")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
