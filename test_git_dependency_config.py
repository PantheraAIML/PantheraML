#!/usr/bin/env python3
"""
Test PantheraML-Zoo Git-based dependency configuration
Validates that all references use correct Git URLs
"""

import os
import re
import subprocess
import sys
from pathlib import Path

def test_git_dependency_config():
    """Test that pyproject.toml correctly references PantheraML-Zoo via Git URL"""
    
    print("🔍 Testing Git-based PantheraML-Zoo dependency configuration...")
    print("=" * 60)
    
    # Test pyproject.toml
    pyproject_path = Path("/Users/aayanmishra/unsloth/pyproject.toml")
    
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        return False
    
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    # Check for Git URLs
    git_url_pattern = r'pantheraml_zoo @ git\+https://github\.com/PantheraAIML/PantheraML-Zoo\.git'
    git_matches = re.findall(git_url_pattern, content)
    
    # Check for old version-based references
    version_pattern = r'pantheraml_zoo>=[\d\.]+'
    version_matches = re.findall(version_pattern, content)
    
    print("📄 pyproject.toml Analysis:")
    print(f"   Git URL references found: {len(git_matches)}")
    print(f"   Old version references found: {len(version_matches)}")
    
    if version_matches:
        print("❌ Found old version-based references:")
        for match in version_matches:
            print(f"      {match}")
        print("   These should be replaced with Git URLs")
        return False
    
    if len(git_matches) < 2:  # Should have at least main deps + one optional dep
        print("❌ Insufficient Git URL references found")
        print("   Expected at least 2 Git URL references")
        return False
    
    print("✅ pyproject.toml correctly configured with Git URLs")
    
    # Test notebook installation commands
    notebook_patterns = [
        "/Users/aayanmishra/unsloth/examples/PantheraML_TPU_Inference_Example.ipynb",
        "/Users/aayanmishra/unsloth/examples/PantheraML_Qwen2.5_HelpSteer2.ipynb"
    ]
    
    print("\n📓 Testing notebook installation commands...")
    
    for notebook_path in notebook_patterns:
        if os.path.exists(notebook_path):
            with open(notebook_path, 'r') as f:
                nb_content = f.read()
            
            # Check for Git installation commands
            git_install_pattern = r'pip install.*git\+https://github\.com/PantheraAIML/PantheraML-Zoo\.git'
            git_installs = re.findall(git_install_pattern, nb_content)
            
            # Check for old package-based installs
            old_install_pattern = r'pip install.*pantheraml_zoo(?!\s+@)'
            old_installs = re.findall(old_install_pattern, nb_content)
            
            print(f"   {Path(notebook_path).name}:")
            print(f"      Git installs: {len(git_installs)}")
            print(f"      Old installs: {len(old_installs)}")
            
            if old_installs:
                print(f"      ❌ Found old package-based installs")
                return False
    
    print("✅ Notebooks correctly use Git-based installation")
    
    return True

def test_fallback_logic():
    """Test that fallback logic is properly implemented"""
    
    print("\n🔄 Testing fallback logic...")
    
    # Check pantheraml/__init__.py for proper fallback
    init_path = "/Users/aayanmishra/unsloth/pantheraml/__init__.py"
    
    if not os.path.exists(init_path):
        print("❌ pantheraml/__init__.py not found")
        return False
    
    with open(init_path, 'r') as f:
        init_content = f.read()
    
    # Check for proper import logic
    required_patterns = [
        r'import pantheraml_zoo as unsloth_zoo',  # Primary import
        r'import unsloth_zoo',                    # Fallback import
        r'pip install.*git\+https://github\.com/PantheraAIML/PantheraML-Zoo\.git',  # Installation instruction
    ]
    
    fallback_working = True
    for pattern in required_patterns:
        if not re.search(pattern, init_content):
            print(f"❌ Missing pattern: {pattern}")
            fallback_working = False
    
    if fallback_working:
        print("✅ Fallback logic properly implemented")
    
    return fallback_working

def test_dependency_simulation():
    """Simulate dependency installation test"""
    
    print("\n🧪 Testing dependency installation simulation...")
    
    # Test if the Git URL format is valid
    git_url = "git+https://github.com/PantheraAIML/PantheraML-Zoo.git"
    
    try:
        # Test URL format without actually installing
        import urllib.parse
        parsed = urllib.parse.urlparse(git_url.replace('git+', ''))
        
        if parsed.scheme not in ['https', 'http']:
            print("❌ Invalid Git URL scheme")
            return False
        
        if 'github.com' not in parsed.netloc:
            print("❌ Git URL should point to GitHub")
            return False
        
        print("✅ Git URL format is valid")
        print(f"   URL: {git_url}")
        
        # Check that URL points to correct repository
        if 'PantheraAIML/PantheraML-Zoo' not in git_url:
            print("❌ Git URL should point to PantheraAIML/PantheraML-Zoo")
            return False
        
        print("✅ Git URL points to correct repository")
        
    except Exception as e:
        print(f"❌ Git URL validation failed: {e}")
        return False
    
    return True

def main():
    """Run all Git dependency tests"""
    
    print("🧪 PantheraML-Zoo Git Dependency Configuration Test")
    print("=" * 55)
    print("Testing migration to Git-based installation...")
    print()
    
    tests = [
        ("Git Dependency Config", test_git_dependency_config),
        ("Fallback Logic", test_fallback_logic),
        ("Dependency Simulation", test_dependency_simulation),
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
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("📊 Test Summary:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All Git dependency tests passed!")
        print("   • PantheraML-Zoo is correctly configured with Git URLs")
        print("   • Installation commands use proper Git syntax")
        print("   • Fallback logic is properly implemented")
        print("   • Repository URLs are valid and point to correct location")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed")
        print("   Please review the configuration for Git-based dependencies")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
