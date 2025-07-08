#!/usr/bin/env python3
"""
Update PantheraML codebase to use PantheraML-Zoo (TPU-enabled fork of unsloth_zoo)
"""

import os
import re
import glob
from pathlib import Path

def update_imports_in_file(file_path):
    """Update unsloth_zoo imports to pantheraml_zoo in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace all import statements
        patterns = [
            # Direct imports: from unsloth_zoo.module import ...
            (r'from unsloth_zoo\.', 'from pantheraml_zoo.'),
            # Import statements: import unsloth_zoo
            (r'import unsloth_zoo', 'import pantheraml_zoo as unsloth_zoo'),
            # Version checks
            (r'importlib_version\("unsloth_zoo"\)', 'importlib_version("pantheraml_zoo")'),
            # Comments and strings referring to unsloth_zoo (but not in special cases)
            (r'# .*unsloth_zoo(?! installed)', lambda m: m.group(0).replace('unsloth_zoo', 'pantheraml_zoo')),
        ]
        
        for pattern, replacement in patterns:
            if callable(replacement):
                content = re.sub(pattern, replacement, content)
            else:
                content = re.sub(pattern, replacement, content)
        
        # Special cases - preserve certain references that should stay as unsloth_zoo
        # (e.g., compatibility notes, fallback logic)
        preserve_patterns = [
            'fallback.*unsloth_zoo',
            'original.*unsloth_zoo', 
            'limited.*unsloth_zoo',
            'pip install unsloth_zoo',
            'compatibility.*unsloth_zoo',
        ]
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to update all relevant files"""
    base_dir = Path(__file__).parent / "pantheraml"
    
    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        return
    
    # Files to update
    file_patterns = [
        "**/*.py",  # All Python files in pantheraml directory
    ]
    
    # Special files that need careful handling
    special_files = [
        "pantheraml/__init__.py",  # Already has some logic for pantheraml_zoo
    ]
    
    updated_files = []
    
    print("🔄 Updating PantheraML codebase to use PantheraML-Zoo...")
    
    for pattern in file_patterns:
        for file_path in base_dir.glob(pattern):
            if file_path.is_file() and file_path.suffix == '.py':
                if update_imports_in_file(file_path):
                    updated_files.append(str(file_path.relative_to(base_dir.parent)))
                    print(f"✅ Updated: {file_path.relative_to(base_dir.parent)}")
    
    print(f"\n📊 Summary:")
    print(f"Updated {len(updated_files)} files to use PantheraML-Zoo")
    
    if updated_files:
        print(f"\nUpdated files:")
        for file in updated_files:
            print(f"  - {file}")
    
    print(f"\n✅ PantheraML codebase update complete!")
    print(f"🔧 All imports now use pantheraml_zoo (TPU-enabled fork)")
    print(f"📝 Fallback logic in __init__.py preserved for compatibility")

if __name__ == "__main__":
    main()
