#!/usr/bin/env python3
"""
Test script to verify the syntax and structure of the helpsteer2 pipeline
without actually importing PantheraML (which requires GPU).
"""

import ast
import sys

def test_syntax_and_structure():
    """Test the syntax and structure of the helpsteer2 pipeline"""
    
    # Test 1: Check if the file has valid Python syntax
    try:
        with open('examples/helpsteer2_complete_pipeline.py', 'r') as f:
            source = f.read()
        
        ast.parse(source)
        print("✅ Python syntax is valid")
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    
    # Test 2: Check for required function definitions
    tree = ast.parse(source)
    
    function_names = []
    class_names = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            class_names.append(node.name)
    
    required_functions = [
        'setup_multi_gpu',
        'setup_model_and_tokenizer', 
        'prepare_dataset',
        'train_model',
        'save_trained_model',
        'load_trained_model_for_inference',
        'run_inference_examples',
        'main',
        'run_kaggle_pipeline',
        'kaggle_quick_test'
    ]
    
    for func in required_functions:
        if func in function_names:
            print(f"✅ Function '{func}' found")
        else:
            print(f"❌ Function '{func}' missing")
            return False
    
    # Test 3: Check for fallback function definitions
    fallback_functions = [
        '_fallback_setup_multi_gpu',
        '_fallback_is_multi_gpu_available',
        '_fallback_get_world_size',
        '_fallback_get_rank',
        '_fallback_is_main_process',
        '_fallback_cleanup_distributed'
    ]
    
    for func in fallback_functions:
        if func in function_names:
            print(f"✅ Fallback function '{func}' found")
        else:
            print(f"❌ Fallback function '{func}' missing")
            return False
    
    # Test 4: Check for proper import structure (look for specific strings)
    if 'from pantheraml import' in source:
        print("✅ PantheraML imports found")
    else:
        print("❌ PantheraML imports missing")
        return False
    
    if 'PANTHERAML_DISTRIBUTED_AVAILABLE' in source:
        print("✅ Distributed availability check found")
    else:
        print("❌ Distributed availability check missing")
        return False
    
    # Test 5: Check for CLI argument parsing
    if 'argparse' in source and 'ArgumentParser' in source:
        print("✅ CLI argument parsing found")
    else:
        print("❌ CLI argument parsing missing")
        return False
    
    print("\n🎯 All syntax and structure tests passed!")
    return True

if __name__ == "__main__":
    success = test_syntax_and_structure()
    sys.exit(0 if success else 1)
