#!/usr/bin/env python3
"""
PantheraML Benchmarking Example

This script demonstrates how to use PantheraML's built-in benchmarking capabilities
to evaluate model performance on standard datasets like MMLU, HellaSwag, and ARC.

Usage:
    python examples/benchmarking_example.py
"""

import torch
from datetime import datetime

def main():
    print("🦥 PantheraML Benchmarking Example")
    print("=" * 60)
    print("   Built on Unsloth's foundation with enhanced benchmarking")
    print()
    
    # Import PantheraML with benchmarking
    try:
        from pantheraml import (
            FastLanguageModel,
            benchmark_mmlu,
            benchmark_hellaswag,
            benchmark_arc,
            PantheraBench
        )
        print("✅ PantheraML benchmarking imports successful")
    except ImportError as e:
        print(f"❌ Failed to import PantheraML: {e}")
        print("   Make sure you have pantheraml installed and are running on GPU/TPU")
        return

    # Example 1: Load a model for benchmarking
    print("\n🤖 Loading model for benchmarking...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="microsoft/DialoGPT-medium",  # Small model for demo
            max_seq_length=2048,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Reduce memory usage
        )
        print(f"✅ Model loaded: {model.config.name_or_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   This is expected on CPU-only systems (like macOS)")
        print("   The benchmarking code is ready for GPU/TPU environments!")
        return

    # Example 2: Quick MMLU benchmark (your requested syntax)
    print("\n📊 Example 1: Quick MMLU Benchmark")
    print("-" * 40)
    
    try:
        # This is the exact syntax you requested!
        result = benchmark_mmlu(model, tokenizer, export=True, max_samples=10)
        
        print(f"🎯 MMLU Results:")
        print(f"   Accuracy: {result.accuracy:.2%}")
        print(f"   Questions: {result.correct_answers}/{result.total_questions}")
        print(f"   Time: {result.execution_time:.2f}s")
        print(f"   Exported: Yes")
        
    except Exception as e:
        print(f"⚠️  Benchmark failed: {e}")

    # Example 3: Using PantheraBench class (comprehensive)
    print("\n📊 Example 2: Comprehensive Benchmark Suite")
    print("-" * 40)
    
    try:
        # Create benchmark suite
        bench = PantheraBench(model, tokenizer)
        
        # Run individual benchmarks
        print("Running MMLU (sample)...")
        mmlu_result = bench.mmlu(max_samples=5, export=True)
        
        print("Running HellaSwag (sample)...")
        hellaswag_result = bench.hellaswag(max_samples=5, export=True)
        
        print("Running ARC-Challenge (sample)...")
        arc_result = bench.arc_challenge(max_samples=5, export=True)
        
        # Or run the full suite
        print("\nRunning comprehensive benchmark suite...")
        all_results = bench.run_suite(
            benchmarks=["mmlu", "hellaswag", "arc_challenge"],
            export=True
        )
        
        print("\n🏆 Final Results Summary:")
        for name, result in all_results.items():
            print(f"   {name.upper()}: {result.accuracy:.2%}")
        
    except Exception as e:
        print(f"⚠️  Benchmark suite failed: {e}")

    # Example 4: Custom benchmark configuration
    print("\n📊 Example 3: Custom Configuration")
    print("-" * 40)
    
    try:
        # MMLU with specific subjects
        from pantheraml import MMLUBenchmark
        
        mmlu_math = MMLUBenchmark(
            model, 
            tokenizer, 
            subjects=["abstract_algebra", "elementary_mathematics"]
        )
        
        math_result = mmlu_math.run(
            max_samples=3,  # Small sample for demo
            export=True,
            export_path="mmlu_math_results"
        )
        
        print(f"🧮 Math-focused MMLU: {math_result.accuracy:.2%}")
        
    except Exception as e:
        print(f"⚠️  Custom benchmark failed: {e}")

    print("\n" + "=" * 60)
    print("📊 Benchmarking Examples Complete!")
    print("\n💡 Usage Patterns:")
    print("   # Quick benchmark:")
    print("   result = benchmark_mmlu(model, tokenizer, export=True)")
    print()
    print("   # Comprehensive suite:")
    print("   bench = PantheraBench(model, tokenizer)")
    print("   results = bench.run_suite(export=True)")
    print()
    print("   # Custom configuration:")
    print("   mmlu = MMLUBenchmark(model, tokenizer, subjects=['math'])")
    print("   result = mmlu.run(export=True)")

def demo_cpu_safe():
    """Demo that works even on CPU-only systems."""
    print("🧪 PantheraML Benchmarking - CPU Safe Demo")
    print("=" * 50)
    
    try:
        # These imports should work even without GPU
        from pantheraml.benchmarks import (
            BenchmarkResult,
            MMLUBenchmark, 
            HellaSwagBenchmark,
            ARCBenchmark,
            PantheraBench
        )
        print("✅ Benchmark classes imported successfully")
        
        # Show what's available
        print("\n📊 Available Benchmarks:")
        print("   • MMLU (Massive Multitask Language Understanding)")
        print("   • HellaSwag (Commonsense Reasoning)")
        print("   • ARC (AI2 Reasoning Challenge)")
        print("   • Custom benchmark support")
        
        print("\n🔧 Benchmark Features:")
        print("   • Automatic result export (JSON/CSV)")
        print("   • Device information tracking")
        print("   • Execution time measurement")
        print("   • Configurable sample sizes")
        print("   • Multi-GPU and TPU support")
        
        print("\n🚀 Ready for GPU/TPU benchmarking!")
        
    except Exception as e:
        print(f"❌ Failed to import benchmarking: {e}")

if __name__ == "__main__":
    # Try full demo first, fall back to CPU-safe demo
    try:
        main()
    except Exception as e:
        print(f"\n⚠️  Full demo failed (expected on CPU-only): {e}")
        print("\nFalling back to CPU-safe demonstration...\n")
        demo_cpu_safe()
