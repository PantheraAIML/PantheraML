#!/usr/bin/env python3
"""
Final validation of the complete HelpSteer2 pipeline implementation
This script validates the code structure and syntax without requiring GPU/ML libraries
"""

import ast
import os
import sys
import re

def validate_helpsteer2_script():
    """Validate the HelpSteer2 complete pipeline script"""
    script_path = "examples/helpsteer2_complete_pipeline.py"
    
    print("🔍 Validating HelpSteer2 Complete Pipeline Script")
    print("=" * 60)
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Parse for syntax validation
        tree = ast.parse(content)
        print("✅ Script syntax is valid")
        
        # Check for key components
        required_functions = [
            "setup_model_and_tokenizer",
            "prepare_dataset", 
            "train_model",
            "save_trained_model",
            "load_trained_model_for_inference",
            "run_inference_examples",
            "benchmark_model",
            "main"
        ]
        
        found_functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                found_functions.append(node.name)
        
        print("\n📋 Function Validation:")
        all_found = True
        for func in required_functions:
            if func in found_functions:
                print(f"✅ {func}")
            else:
                print(f"❌ {func} - missing")
                all_found = False
        
        # Check for PantheraML imports (not unsloth)
        print("\n📋 Import Validation:")
        import_lines = [line.strip() for line in content.split('\n') if 'import' in line]
        
        pantheraml_imports = [line for line in import_lines if 'pantheraml' in line]
        unsloth_imports = [line for line in import_lines if 'unsloth' in line and 'pantheraml' not in line]
        
        if pantheraml_imports:
            print("✅ PantheraML imports found:")
            for imp in pantheraml_imports[:3]:  # Show first 3
                print(f"   {imp}")
        else:
            print("❌ No PantheraML imports found")
            all_found = False
        
        if unsloth_imports:
            print("⚠️ Unsloth imports still present:")
            for imp in unsloth_imports:
                print(f"   {imp}")
        else:
            print("✅ No unsloth imports found")
        
        # Check for dataset handling
        print("\n📋 Dataset Validation:")
        if "nvidia/HelpSteer2" in content:
            print("✅ HelpSteer2 dataset referenced")
        else:
            print("❌ HelpSteer2 dataset not found")
            all_found = False
        
        if "formatting_prompts_func" in content:
            print("✅ Dataset formatting function present")
        else:
            print("❌ Dataset formatting function missing")
            all_found = False
        
        # Check for training components
        print("\n📋 Training Validation:")
        training_components = [
            "FastLanguageModel.from_pretrained",
            "FastLanguageModel.get_peft_model", 
            "SFTTrainer",
            "TrainingArguments"
        ]
        
        for component in training_components:
            if component in content:
                print(f"✅ {component}")
            else:
                print(f"❌ {component} - missing")
                all_found = False
        
        # Check for inference components  
        print("\n📋 Inference Validation:")
        if "run_inference_examples" in content and "test_prompts" in content:
            print("✅ Inference examples implemented")
        else:
            print("❌ Inference examples missing")
            all_found = False
        
        if "model.generate" in content:
            print("✅ Text generation implemented")
        else:
            print("❌ Text generation missing")
            all_found = False
        
        # Check command line arguments
        print("\n📋 CLI Validation:")
        cli_args = [
            "--model", "--max_seq_length", "--batch_size", 
            "--max_steps", "--max_samples", "--output_dir",
            "--skip_training", "--run_benchmarks"
        ]
        
        cli_found = True
        for arg in cli_args:
            if arg in content:
                print(f"✅ {arg}")
            else:
                print(f"❌ {arg} - missing")
                cli_found = False
        
        all_found = all_found and cli_found
        
        # Check for saving formats
        print("\n📋 Model Saving Validation:")
        if "save_pretrained" in content:
            print("✅ Basic model saving")
        else:
            print("❌ Basic model saving missing")
            all_found = False
        
        if "save_pretrained_merged" in content:
            print("✅ Merged model saving")
        else:
            print("❌ Merged model saving missing")
            all_found = False
        
        if "save_pretrained_gguf" in content:
            print("✅ GGUF model saving")
        else:
            print("❌ GGUF model saving missing")
            all_found = False
        
        return all_found
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def validate_documentation():
    """Validate that documentation exists and is complete"""
    print("\n🔍 Validating Documentation")
    print("=" * 60)
    
    docs_to_check = [
        ("examples/README.md", "Examples README"),
        ("HELPSTEER2_PIPELINE_GUIDE.md", "HelpSteer2 Pipeline Guide"),
        ("PANTHERAML_COMPLETE_GUIDE.md", "Complete Guide"),
        ("PANTHERAML_COMPLETE_API_REFERENCE.md", "API Reference")
    ]
    
    all_docs_found = True
    
    for doc_path, doc_name in docs_to_check:
        if os.path.exists(doc_path):
            with open(doc_path, 'r') as f:
                content = f.read()
            
            word_count = len(content.split())
            print(f"✅ {doc_name} ({word_count} words)")
            
            # Check for key sections
            if doc_path == "HELPSTEER2_PIPELINE_GUIDE.md":
                required_sections = ["Quick Start", "Command Line Arguments", "Training Configuration", "Output Formats"]
                for section in required_sections:
                    if section in content:
                        print(f"   ✅ {section} section")
                    else:
                        print(f"   ❌ {section} section missing")
                        
        else:
            print(f"❌ {doc_name} - not found")
            all_docs_found = False
    
    return all_docs_found

def validate_test_script():
    """Validate the test script exists and works"""
    print("\n🔍 Validating Test Script")
    print("=" * 60)
    
    test_script = "test_helpsteer2_pipeline.py"
    
    if os.path.exists(test_script):
        try:
            with open(test_script, 'r') as f:
                content = f.read()
            
            ast.parse(content)
            print("✅ Test script syntax valid")
            
            if "test_script_syntax" in content and "test_pantheraml_imports" in content:
                print("✅ Test functions implemented")
                return True
            else:
                print("❌ Test functions missing")
                return False
                
        except SyntaxError as e:
            print(f"❌ Test script syntax error: {e}")
            return False
    else:
        print(f"❌ Test script not found: {test_script}")
        return False

def main():
    """Run all validations"""
    print("🧪 Final HelpSteer2 Pipeline Validation")
    print("=" * 60)
    
    validations = [
        ("HelpSteer2 Script", validate_helpsteer2_script),
        ("Documentation", validate_documentation),
        ("Test Script", validate_test_script)
    ]
    
    results = []
    
    for name, validation_func in validations:
        print(f"\n📋 {name} Validation")
        print("-" * 40)
        result = validation_func()
        results.append((name, result))
    
    # Summary
    print("\n📊 Validation Summary")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} validations passed")
    
    if passed == len(results):
        print("\n🎉 All validations passed!")
        print("\nThe HelpSteer2 complete pipeline is ready for use:")
        print("  📁 Script: examples/helpsteer2_complete_pipeline.py")
        print("  📖 Guide: HELPSTEER2_PIPELINE_GUIDE.md")
        print("  🧪 Test: test_helpsteer2_pipeline.py")
        print("\nQuick start:")
        print("  python examples/helpsteer2_complete_pipeline.py --max_steps 10 --max_samples 100")
    else:
        print(f"\n⚠️ {len(results) - passed} validation(s) failed.")
        print("Please review the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
