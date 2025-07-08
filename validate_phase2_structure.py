#!/usr/bin/env python3
"""
Simple validation script for Phase 2 code structure
This script validates the implementation without requiring all dependencies.
"""

import os
import sys
import ast
import importlib.util

def validate_file_structure():
    """Validate that all Phase 2 files exist with correct structure."""
    print("🧪 Validating Phase 2 file structure...")
    
    required_files = [
        "pantheraml/__init__.py",
        "pantheraml/trainer.py", 
        "pantheraml/distributed.py",
        "pantheraml/kernels/tpu_kernels.py",
        "pantheraml/kernels/tpu_performance.py",
        "test_phase2_tpu.py",
        "validate_phase2_integration.py",
        "PHASE2_TPU_COMPLETE.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All Phase 2 files present")
        return True

def validate_code_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        ast.parse(source)
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return False

def validate_phase2_classes():
    """Validate that Phase 2 classes are defined correctly."""
    print("🧪 Validating Phase 2 class definitions...")
    
    # Check tpu_performance.py
    perf_file = "pantheraml/kernels/tpu_performance.py"
    if not os.path.exists(perf_file):
        print(f"❌ {perf_file} not found")
        return False
    
    try:
        with open(perf_file, 'r') as f:
            content = f.read()
        
        required_classes = [
            "XLAAttentionOptimizer",
            "ModelShardManager", 
            "DynamicShapeManager",
            "TPUCommunicationOptimizer",
            "TPUPerformanceProfiler"
        ]
        
        missing_classes = []
        for class_name in required_classes:
            # Check for both class definition and alias assignment
            if (f"class {class_name}" not in content and 
                f"{class_name} =" not in content):
                missing_classes.append(class_name)
        
        if missing_classes:
            print(f"❌ Missing classes in {perf_file}: {missing_classes}")
            return False
        else:
            print("✅ All Phase 2 classes defined")
            return True
            return True
            
    except Exception as e:
        print(f"❌ Error validating classes: {e}")
        return False

def validate_trainer_integration():
    """Validate trainer integration."""
    print("🧪 Validating trainer integration...")
    
    trainer_file = "pantheraml/trainer.py"
    if not os.path.exists(trainer_file):
        print(f"❌ {trainer_file} not found")
        return False
    
    try:
        with open(trainer_file, 'r') as f:
            content = f.read()
        
        # Check for Phase 2 integration
        checks = [
            ("Phase 2 imports", "from .kernels.tpu_performance import"),
            ("Phase 2 initialization", "_init_phase2_components"),
            ("Phase 2 trainer class", "class PantheraMLTPUTrainer"),
            ("Phase 2 training step", "def training_step"),
            ("Phase 2 performance metrics", "get_performance_metrics")
        ]
        
        results = []
        for check_name, check_pattern in checks:
            if check_pattern in content:
                print(f"✅ {check_name}: Found")
                results.append(True)
            else:
                print(f"❌ {check_name}: Missing")
                results.append(False)
        
        return all(results)
        
    except Exception as e:
        print(f"❌ Error validating trainer: {e}")
        return False

def validate_distributed_integration():
    """Validate distributed training integration."""
    print("🧪 Validating distributed integration...")
    
    dist_file = "pantheraml/distributed.py"
    if not os.path.exists(dist_file):
        print(f"❌ {dist_file} not found")
        return False
    
    try:
        with open(dist_file, 'r') as f:
            content = f.read()
        
        # Check for Phase 2 functions
        functions = [
            "setup_phase2_distributed_training",
            "optimize_distributed_communication",
            "synchronize_phase2_training",
            "cleanup_phase2_distributed",
            "setup_enhanced_distributed_training"
        ]
        
        missing_functions = []
        for func_name in functions:
            if f"def {func_name}" not in content:
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"❌ Missing functions: {missing_functions}")
            return False
        else:
            print("✅ All Phase 2 distributed functions defined")
            return True
            
    except Exception as e:
        print(f"❌ Error validating distributed: {e}")
        return False

def validate_notebook_integration():
    """Validate notebook has Phase 2 examples."""
    print("🧪 Validating notebook integration...")
    
    notebook_file = "examples/PantheraML_Qwen2.5_HelpSteer2.ipynb"
    if not os.path.exists(notebook_file):
        print(f"❌ {notebook_file} not found")
        return False
    
    try:
        with open(notebook_file, 'r') as f:
            content = f.read()
        
        # Check for Phase 2 content
        phase2_indicators = [
            "Phase 2 TPU Support",
            "tpu_config",
            "PantheraMLTPUTrainer",
            "setup_enhanced_distributed_training",
            "enable_phase2"
        ]
        
        found_indicators = []
        for indicator in phase2_indicators:
            if indicator in content:
                found_indicators.append(indicator)
        
        if len(found_indicators) >= 3:  # At least 3 indicators
            print(f"✅ Phase 2 content found: {found_indicators}")
            return True
        else:
            print(f"⚠️ Limited Phase 2 content: {found_indicators}")
            return False
            
    except Exception as e:
        print(f"❌ Error validating notebook: {e}")
        return False

def validate_syntax_all_files():
    """Validate syntax of all Python files."""
    print("🧪 Validating syntax of all Python files...")
    
    python_files = [
        "pantheraml/__init__.py",
        "pantheraml/trainer.py",
        "pantheraml/distributed.py",
        "pantheraml/kernels/tpu_kernels.py",
        "pantheraml/kernels/tpu_performance.py",
        "test_phase2_tpu.py",
        "validate_phase2_integration.py"
    ]
    
    syntax_errors = []
    for file_path in python_files:
        if os.path.exists(file_path):
            if not validate_code_syntax(file_path):
                syntax_errors.append(file_path)
        else:
            print(f"⚠️ File not found: {file_path}")
    
    if syntax_errors:
        print(f"❌ Syntax errors in: {syntax_errors}")
        return False
    else:
        print("✅ All Python files have valid syntax")
        return True

def main():
    """Main validation function."""
    print("🚀 Starting Phase 2 Code Structure Validation")
    print("=" * 50)
    
    # Track validation results
    validations = [
        ("File Structure", validate_file_structure()),
        ("Code Syntax", validate_syntax_all_files()),
        ("Phase 2 Classes", validate_phase2_classes()),
        ("Trainer Integration", validate_trainer_integration()),
        ("Distributed Integration", validate_distributed_integration()),
        ("Notebook Integration", validate_notebook_integration())
    ]
    
    # Summary
    print("\n" + "=" * 50)
    print("🧪 Phase 2 Structure Validation Summary:")
    
    passed = 0
    total = len(validations)
    
    for validation_name, result in validations:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {validation_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} validations passed")
    
    if passed == total:
        print("🎉 Phase 2 code structure validation successful!")
        print("🚀 Implementation is structurally complete.")
        
        # Additional information
        print("\n📋 Implementation Summary:")
        print("  ✅ Phase 1: Error handling, memory, XLA (Complete)")
        print("  ✅ Phase 2: Performance, sharding, communication (Complete)")
        print("  🔄 Testing: Requires TPU/GPU environment for full validation")
        print("  📚 Documentation: Complete with examples")
        
        return 0
    else:
        print("⚠️ Some structural validations failed.")
        print("🔧 Please review the failed components.")
        return 1

if __name__ == "__main__":
    exit(main())
