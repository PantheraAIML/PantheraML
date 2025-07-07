# PantheraML Built-in Benchmarking Guide

## 🧪 Built-in Benchmarking System

PantheraML now includes a comprehensive benchmarking system that supports popular LLM evaluation datasets with automatic result export and device performance tracking.

## 📊 Supported Benchmarks

- **MMLU** (Massive Multitask Language Understanding) - 57 academic subjects
- **HellaSwag** - Commonsense reasoning
- **ARC** (AI2 Reasoning Challenge) - Scientific reasoning
- **Custom benchmarks** - Extensible framework

## 🚀 Quick Start (Your Requested Syntax)

```python
from pantheraml import FastLanguageModel, benchmark_mmlu

# Load your model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/DialoGPT-medium",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Run MMLU benchmark with automatic export
result = benchmark_mmlu(model, tokenizer, export=True)

print(f"MMLU Accuracy: {result.accuracy:.2%}")
print(f"Results exported to: mmlu_results_{timestamp}.json")
```

## 📈 Comprehensive Benchmarking

### Method 1: Individual Benchmarks
```python
from pantheraml import benchmark_mmlu, benchmark_hellaswag, benchmark_arc

# Run individual benchmarks
mmlu_result = benchmark_mmlu(model, tokenizer, export=True)
hellaswag_result = benchmark_hellaswag(model, tokenizer, export=True)
arc_result = benchmark_arc(model, tokenizer, export=True)

print(f"MMLU: {mmlu_result.accuracy:.2%}")
print(f"HellaSwag: {hellaswag_result.accuracy:.2%}")
print(f"ARC: {arc_result.accuracy:.2%}")
```

### Method 2: Benchmark Suite
```python
from pantheraml import PantheraBench

# Create benchmark suite
bench = PantheraBench(model, tokenizer)

# Run all benchmarks with export
results = bench.run_suite(export=True)

# Results automatically saved to timestamped directory
for name, result in results.items():
    print(f"{name.upper()}: {result.accuracy:.2%}")
```

### Method 3: Custom Configuration
```python
from pantheraml import MMLUBenchmark, HellaSwagBenchmark

# MMLU with specific subjects
mmlu_math = MMLUBenchmark(
    model, 
    tokenizer, 
    subjects=["abstract_algebra", "elementary_mathematics", "high_school_mathematics"]
)
math_result = mmlu_math.run(export=True, export_path="math_benchmark")

# HellaSwag with sample limit
hellaswag = HellaSwagBenchmark(model, tokenizer)
quick_result = hellaswag.run(max_samples=100, export=True)
```

## 🔧 Benchmark Configuration Options

### MMLU Benchmark
```python
benchmark_mmlu(
    model, 
    tokenizer,
    num_shots=0,           # Few-shot examples (0 for zero-shot)
    max_samples=None,      # Limit questions per subject (None = all)
    subjects="all",        # "all" or list of specific subjects
    export=True,           # Export results to JSON/CSV
    export_path=None       # Custom export path (auto-generated if None)
)
```

### HellaSwag Benchmark
```python
benchmark_hellaswag(
    model,
    tokenizer, 
    max_samples=None,      # Limit total questions (None = all)
    export=True,
    export_path=None
)
```

### ARC Benchmark
```python
benchmark_arc(
    model,
    tokenizer,
    challenge_set="ARC-Challenge",  # "ARC-Challenge" or "ARC-Easy"
    max_samples=None,
    export=True,
    export_path=None
)
```

## 📄 Result Format

Each benchmark returns a `BenchmarkResult` object with:

```python
@dataclass
class BenchmarkResult:
    benchmark_name: str      # "MMLU", "HellaSwag", etc.
    model_name: str         # Model identifier
    accuracy: float         # Overall accuracy (0.0 to 1.0)
    total_questions: int    # Number of questions evaluated
    correct_answers: int    # Number of correct answers
    execution_time: float   # Time taken in seconds
    timestamp: str          # ISO timestamp
    device_info: Dict       # GPU/TPU/CPU information
    config: Dict           # Benchmark configuration used
```

## 💾 Export Formats

Results are automatically exported in multiple formats:

### JSON Export
```json
{
  "benchmark_name": "MMLU",
  "model_name": "microsoft/DialoGPT-medium",
  "accuracy": 0.726,
  "total_questions": 14042,
  "correct_answers": 10194,
  "execution_time": 2847.32,
  "timestamp": "2025-01-07T15:30:45",
  "device_info": {
    "device_type": "cuda:0",
    "gpu_name": "NVIDIA A100",
    "gpu_memory": 42949672960
  },
  "config": {
    "num_shots": 0,
    "max_samples": null,
    "subjects": "all"
  }
}
```

### CSV Export
Results are also saved as CSV for easy analysis in spreadsheets or data science tools.

## 🚀 Multi-GPU and TPU Support

The benchmarking system automatically supports:

- **Multi-GPU setups**: Distributes evaluation across available GPUs
- **TPU support**: Experimental TPU evaluation with torch_xla
- **Mixed precision**: Automatic optimization for faster evaluation
- **Memory management**: Efficient memory usage for large models

```python
# Multi-GPU benchmarking (automatic)
from pantheraml import FastLanguageModel, benchmark_mmlu, setup_multi_gpu

if torch.cuda.device_count() > 1:
    setup_multi_gpu()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/DialoGPT-large",
    use_multi_gpu=True,
    auto_device_map=True
)

result = benchmark_mmlu(model, tokenizer, export=True)
```

## 📊 Benchmark Suite with Export

```python
from pantheraml import PantheraBench

# Comprehensive evaluation
bench = PantheraBench(model, tokenizer)

# Run all benchmarks and export to organized directory
results = bench.run_suite(
    benchmarks=["mmlu", "hellaswag", "arc_challenge", "arc_easy"],
    export=True,
    export_dir="my_model_evaluation_2025_01_07"
)

# Directory structure:
# my_model_evaluation_2025_01_07/
# ├── mmlu_results.json
# ├── mmlu_results.csv
# ├── hellaswag_results.json
# ├── hellaswag_results.csv
# ├── arc_challenge_results.json
# ├── arc_challenge_results.csv
# ├── arc_easy_results.json
# ├── arc_easy_results.csv
# └── summary.json
```

## 🔍 Advanced Usage

### Custom Benchmark Development
```python
from pantheraml.benchmarks import BaseBenchmark

class MyCustomBenchmark(BaseBenchmark):
    def run(self, **kwargs):
        # Implement your custom benchmark logic
        # Return BenchmarkResult object
        pass
```

### Progress Monitoring
```python
# Benchmarks show progress automatically
# Progress: 1000/14042 (7.1%)
# Progress: 2000/14042 (14.2%) 
# Progress: 3000/14042 (21.4%)
```

### Memory Optimization
```python
# Automatic memory optimization during benchmarking
# Works with quantized models, gradient checkpointing, etc.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/DialoGPT-large",
    load_in_4bit=True,           # Quantization
    use_gradient_checkpointing=True,  # Memory efficiency
)

result = benchmark_mmlu(model, tokenizer, export=True)
# Benchmarking respects all model optimizations
```

## 🎯 Real-World Example

```python
#!/usr/bin/env python3
"""
Complete model evaluation pipeline with PantheraML
"""
from pantheraml import FastLanguageModel, PantheraBench
import torch

def evaluate_model(model_name: str):
    print(f"🧪 Evaluating {model_name}")
    
    # Load model with optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        use_multi_gpu=torch.cuda.device_count() > 1
    )
    
    # Create benchmark suite
    bench = PantheraBench(model, tokenizer)
    
    # Run comprehensive evaluation
    results = bench.run_suite(export=True)
    
    # Print summary
    print(f"\n📊 {model_name} Results:")
    for name, result in results.items():
        print(f"   {name.upper()}: {result.accuracy:.2%}")
    
    return results

# Evaluate multiple models
models = [
    "microsoft/DialoGPT-medium",
    "microsoft/DialoGPT-large", 
    "facebook/opt-1.3b"
]

for model_name in models:
    evaluate_model(model_name)
```

## 🙏 Credits

The benchmarking system is built on PantheraML's foundation, which extends the excellent work of the Unsloth team. All original Unsloth performance optimizations are preserved and enhanced for benchmarking workloads.

**Ready to benchmark your models with PantheraML!** 🚀
