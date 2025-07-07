# Copyright 2023-present Daniel Han-Chen & the PantheraML team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PantheraML Benchmarking Module

This module provides built-in benchmarking capabilities for evaluating models
on standard datasets like MMLU, HellaSwag, ARC, etc.

Built on Unsloth's foundation with enhanced multi-GPU and TPU support.
"""

import os
import json
import time
import torch
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import torch_xla.core.xla_model as xm
    HAS_TPU = True
except ImportError:
    HAS_TPU = False


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    benchmark_name: str
    model_name: str
    accuracy: float
    total_questions: int
    correct_answers: int
    execution_time: float
    timestamp: str
    device_info: Dict[str, Any]
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return asdict(self)
    
    def save_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_csv(self, filepath: str):
        """Save results to CSV file."""
        df = pd.DataFrame([self.to_dict()])
        df.to_csv(filepath, index=False)


class BaseBenchmark:
    """Base class for all benchmarks."""
    
    def __init__(self, model, tokenizer, device=None, batch_size=1):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.batch_size = batch_size
        self.results = []
        
    def _get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device."""
        device_info = {
            'device_type': str(self.device),
            'torch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            device_info.update({
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            })
        
        if HAS_TPU:
            try:
                device_info.update({
                    'tpu_cores': xm.xrt_world_size(),
                    'tpu_device': str(xm.xla_device()),
                })
            except:
                pass
                
        return device_info
    
    def _format_prompt(self, question: str, choices: List[str], subject: str = "") -> str:
        """Format the prompt for multiple choice questions."""
        prompt = f"Subject: {subject}\n\n" if subject else ""
        prompt += f"Question: {question}\n\n"
        
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(choices)]
        for label, choice in zip(choice_labels, choices):
            prompt += f"{label}. {choice}\n"
        
        prompt += "\nAnswer: "
        return prompt
    
    def _get_model_response(self, prompt: str) -> str:
        """Get model response to a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    
    def _extract_answer(self, response: str) -> str:
        """Extract the answer choice from model response."""
        response = response.strip().upper()
        valid_choices = ['A', 'B', 'C', 'D', 'E', 'F']
        
        for choice in valid_choices:
            if response.startswith(choice):
                return choice
        
        # Fallback: return first valid letter found
        for char in response:
            if char in valid_choices:
                return char
        
        return "A"  # Default fallback
    
    def run(self, **kwargs) -> BenchmarkResult:
        """Run the benchmark. To be implemented by subclasses."""
        raise NotImplementedError


class MMLUBenchmark(BaseBenchmark):
    """
    MMLU (Massive Multitask Language Understanding) Benchmark
    
    Tests model performance across 57 academic subjects including
    mathematics, history, computer science, law, and more.
    """
    
    def __init__(self, model, tokenizer, device=None, batch_size=1, subjects=None):
        super().__init__(model, tokenizer, device, batch_size)
        self.subjects = subjects or "all"  # Can specify specific subjects or "all"
        
    def run(self, num_shots=0, max_samples=None, export=False, export_path=None) -> BenchmarkResult:
        """
        Run MMLU benchmark.
        
        Args:
            num_shots: Number of few-shot examples (0 for zero-shot)
            max_samples: Maximum number of samples per subject (None for all)
            export: Whether to export results
            export_path: Path to export results (auto-generated if None)
            
        Returns:
            BenchmarkResult with accuracy and other metrics
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library required for MMLU benchmark. Install with: pip install datasets")
        
        print("🧪 Starting MMLU Benchmark...")
        print(f"   📊 Subjects: {self.subjects}")
        print(f"   🎯 Few-shot examples: {num_shots}")
        print(f"   🔢 Max samples per subject: {max_samples or 'all'}")
        
        start_time = time.time()
        
        # Load MMLU dataset
        dataset = load_dataset("cais/mmlu", "all", split="test")
        
        # Filter subjects if specified
        if self.subjects != "all" and isinstance(self.subjects, list):
            dataset = dataset.filter(lambda x: x["subject"] in self.subjects)
        
        # Limit samples if specified
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        total_questions = len(dataset)
        correct_answers = 0
        
        print(f"   📋 Total questions: {total_questions}")
        print("   🚀 Running evaluation...")
        
        for i, example in enumerate(dataset):
            if i % 100 == 0:
                print(f"   Progress: {i}/{total_questions} ({i/total_questions*100:.1f}%)")
            
            # Format the prompt
            prompt = self._format_prompt(
                question=example["question"],
                choices=example["choices"],
                subject=example["subject"]
            )
            
            # Get model response
            response = self._get_model_response(prompt)
            predicted_answer = self._extract_answer(response)
            
            # Check if correct (MMLU answers are 0-indexed, convert to A-D)
            correct_choice = ['A', 'B', 'C', 'D'][example["answer"]]
            if predicted_answer == correct_choice:
                correct_answers += 1
        
        execution_time = time.time() - start_time
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        # Create result
        result = BenchmarkResult(
            benchmark_name="MMLU",
            model_name=getattr(self.model, 'name_or_path', 'unknown'),
            accuracy=accuracy,
            total_questions=total_questions,
            correct_answers=correct_answers,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            device_info=self._get_device_info(),
            config={
                'num_shots': num_shots,
                'max_samples': max_samples,
                'subjects': self.subjects,
                'batch_size': self.batch_size
            }
        )
        
        print(f"\n🎯 MMLU Results:")
        print(f"   📊 Accuracy: {accuracy:.2%} ({correct_answers}/{total_questions})")
        print(f"   ⏱️  Execution time: {execution_time:.2f}s")
        
        # Export results if requested
        if export:
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"mmlu_results_{timestamp}"
            
            result.save_json(f"{export_path}.json")
            result.save_csv(f"{export_path}.csv")
            print(f"   💾 Results exported to {export_path}.json and {export_path}.csv")
        
        return result


class HellaSwagBenchmark(BaseBenchmark):
    """HellaSwag benchmark for commonsense reasoning."""
    
    def run(self, max_samples=None, export=False, export_path=None) -> BenchmarkResult:
        """Run HellaSwag benchmark."""
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        print("🧪 Starting HellaSwag Benchmark...")
        
        start_time = time.time()
        dataset = load_dataset("hellaswag", split="validation")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        total_questions = len(dataset)
        correct_answers = 0
        
        print(f"   📋 Total questions: {total_questions}")
        
        for i, example in enumerate(dataset):
            if i % 100 == 0:
                print(f"   Progress: {i}/{total_questions} ({i/total_questions*100:.1f}%)")
            
            prompt = self._format_prompt(
                question=f"{example['ctx']} {example['activity_label']}",
                choices=example["endings"]
            )
            
            response = self._get_model_response(prompt)
            predicted_answer = self._extract_answer(response)
            
            correct_choice = ['A', 'B', 'C', 'D'][int(example["label"])]
            if predicted_answer == correct_choice:
                correct_answers += 1
        
        execution_time = time.time() - start_time
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        result = BenchmarkResult(
            benchmark_name="HellaSwag",
            model_name=getattr(self.model, 'name_or_path', 'unknown'),
            accuracy=accuracy,
            total_questions=total_questions,
            correct_answers=correct_answers,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            device_info=self._get_device_info(),
            config={'max_samples': max_samples, 'batch_size': self.batch_size}
        )
        
        print(f"\n🎯 HellaSwag Results:")
        print(f"   📊 Accuracy: {accuracy:.2%} ({correct_answers}/{total_questions})")
        print(f"   ⏱️  Execution time: {execution_time:.2f}s")
        
        if export:
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"hellaswag_results_{timestamp}"
            
            result.save_json(f"{export_path}.json")
            result.save_csv(f"{export_path}.csv")
            print(f"   💾 Results exported to {export_path}.json and {export_path}.csv")
        
        return result


class ARCBenchmark(BaseBenchmark):
    """AI2 Reasoning Challenge (ARC) benchmark."""
    
    def __init__(self, model, tokenizer, device=None, batch_size=1, challenge_set="ARC-Challenge"):
        super().__init__(model, tokenizer, device, batch_size)
        self.challenge_set = challenge_set  # "ARC-Challenge" or "ARC-Easy"
    
    def run(self, max_samples=None, export=False, export_path=None) -> BenchmarkResult:
        """Run ARC benchmark."""
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        print(f"🧪 Starting ARC Benchmark ({self.challenge_set})...")
        
        start_time = time.time()
        dataset = load_dataset("ai2_arc", self.challenge_set, split="test")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        total_questions = len(dataset)
        correct_answers = 0
        
        print(f"   📋 Total questions: {total_questions}")
        
        for i, example in enumerate(dataset):
            if i % 50 == 0:
                print(f"   Progress: {i}/{total_questions} ({i/total_questions*100:.1f}%)")
            
            choices = [choice["text"] for choice in example["choices"]["text"]]
            labels = example["choices"]["label"]
            
            prompt = self._format_prompt(
                question=example["question"],
                choices=choices
            )
            
            response = self._get_model_response(prompt)
            predicted_answer = self._extract_answer(response)
            
            # Find correct choice index
            correct_idx = labels.index(example["answerKey"])
            correct_choice = ['A', 'B', 'C', 'D', 'E'][correct_idx]
            
            if predicted_answer == correct_choice:
                correct_answers += 1
        
        execution_time = time.time() - start_time
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        result = BenchmarkResult(
            benchmark_name=f"ARC-{self.challenge_set}",
            model_name=getattr(self.model, 'name_or_path', 'unknown'),
            accuracy=accuracy,
            total_questions=total_questions,
            correct_answers=correct_answers,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            device_info=self._get_device_info(),
            config={'challenge_set': self.challenge_set, 'max_samples': max_samples}
        )
        
        print(f"\n🎯 ARC ({self.challenge_set}) Results:")
        print(f"   📊 Accuracy: {accuracy:.2%} ({correct_answers}/{total_questions})")
        print(f"   ⏱️  Execution time: {execution_time:.2f}s")
        
        if export:
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"arc_{self.challenge_set.lower()}_{timestamp}"
            
            result.save_json(f"{export_path}.json")
            result.save_csv(f"{export_path}.csv")
            print(f"   💾 Results exported to {export_path}.json and {export_path}.csv")
        
        return result


# Convenience functions for easy access
def benchmark_mmlu(model, tokenizer=None, **kwargs) -> BenchmarkResult:
    """
    Convenience function to run MMLU benchmark.
    
    Usage:
        from pantheraml import benchmark_mmlu
        
        model, tokenizer = FastLanguageModel.from_pretrained(...)
        result = benchmark_mmlu(model, tokenizer, export=True)
    """
    if tokenizer is None:
        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided or model must have a tokenizer attribute")
    
    benchmark = MMLUBenchmark(model, tokenizer)
    return benchmark.run(**kwargs)


def benchmark_hellaswag(model, tokenizer=None, **kwargs) -> BenchmarkResult:
    """Convenience function to run HellaSwag benchmark."""
    if tokenizer is None:
        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided or model must have a tokenizer attribute")
    
    benchmark = HellaSwagBenchmark(model, tokenizer)
    return benchmark.run(**kwargs)


def benchmark_arc(model, tokenizer=None, challenge_set="ARC-Challenge", **kwargs) -> BenchmarkResult:
    """Convenience function to run ARC benchmark."""
    if tokenizer is None:
        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided or model must have a tokenizer attribute")
    
    benchmark = ARCBenchmark(model, tokenizer, challenge_set=challenge_set)
    return benchmark.run(**kwargs)


class PantheraBench:
    """
    Main benchmarking class that provides access to all benchmarks.
    
    Usage:
        from pantheraml import PantheraBench
        
        model, tokenizer = FastLanguageModel.from_pretrained(...)
        bench = PantheraBench(model, tokenizer)
        
        # Run individual benchmarks
        mmlu_result = bench.mmlu(export=True)
        hellaswag_result = bench.hellaswag(export=True)
        
        # Run comprehensive benchmark suite
        all_results = bench.run_suite(export=True)
    """
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer or getattr(model, 'tokenizer', None)
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided or model must have a tokenizer attribute")
    
    def mmlu(self, **kwargs) -> BenchmarkResult:
        """Run MMLU benchmark."""
        benchmark = MMLUBenchmark(self.model, self.tokenizer)
        return benchmark.run(**kwargs)
    
    def hellaswag(self, **kwargs) -> BenchmarkResult:
        """Run HellaSwag benchmark."""
        benchmark = HellaSwagBenchmark(self.model, self.tokenizer)
        return benchmark.run(**kwargs)
    
    def arc_challenge(self, **kwargs) -> BenchmarkResult:
        """Run ARC-Challenge benchmark."""
        benchmark = ARCBenchmark(self.model, self.tokenizer, challenge_set="ARC-Challenge")
        return benchmark.run(**kwargs)
    
    def arc_easy(self, **kwargs) -> BenchmarkResult:
        """Run ARC-Easy benchmark."""
        benchmark = ARCBenchmark(self.model, self.tokenizer, challenge_set="ARC-Easy")
        return benchmark.run(**kwargs)
    
    def run_suite(self, benchmarks=None, export=False, export_dir=None) -> Dict[str, BenchmarkResult]:
        """
        Run a comprehensive benchmark suite.
        
        Args:
            benchmarks: List of benchmark names to run (default: all)
            export: Whether to export results
            export_dir: Directory to export results to
            
        Returns:
            Dictionary of benchmark results
        """
        if benchmarks is None:
            benchmarks = ["mmlu", "hellaswag", "arc_challenge", "arc_easy"]
        
        if export_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"pantheraml_benchmark_suite_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)
        
        results = {}
        
        print("🚀 PantheraML Benchmark Suite")
        print("=" * 50)
        print(f"   🤖 Model: {getattr(self.model, 'name_or_path', 'unknown')}")
        print(f"   📊 Benchmarks: {', '.join(benchmarks)}")
        print(f"   💾 Export: {'Yes' if export else 'No'}")
        if export:
            print(f"   📁 Export directory: {export_dir}")
        print()
        
        for benchmark_name in benchmarks:
            print(f"🔄 Running {benchmark_name.upper()}...")
            
            try:
                if benchmark_name == "mmlu":
                    result = self.mmlu(
                        export=export,
                        export_path=os.path.join(export_dir, "mmlu_results") if export else None
                    )
                elif benchmark_name == "hellaswag":
                    result = self.hellaswag(
                        export=export,
                        export_path=os.path.join(export_dir, "hellaswag_results") if export else None
                    )
                elif benchmark_name == "arc_challenge":
                    result = self.arc_challenge(
                        export=export,
                        export_path=os.path.join(export_dir, "arc_challenge_results") if export else None
                    )
                elif benchmark_name == "arc_easy":
                    result = self.arc_easy(
                        export=export,
                        export_path=os.path.join(export_dir, "arc_easy_results") if export else None
                    )
                else:
                    print(f"   ⚠️  Unknown benchmark: {benchmark_name}")
                    continue
                
                results[benchmark_name] = result
                print(f"   ✅ {benchmark_name.upper()} completed: {result.accuracy:.2%}")
                
            except Exception as e:
                print(f"   ❌ {benchmark_name.upper()} failed: {e}")
        
        # Create summary
        if results and export:
            summary = {
                'model_name': getattr(self.model, 'name_or_path', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'benchmarks': {name: result.accuracy for name, result in results.items()},
                'average_accuracy': sum(r.accuracy for r in results.values()) / len(results)
            }
            
            with open(os.path.join(export_dir, "summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n📊 Benchmark Suite Summary:")
            for name, result in results.items():
                print(f"   {name.upper()}: {result.accuracy:.2%}")
            print(f"   Average: {summary['average_accuracy']:.2%}")
            print(f"   💾 Summary saved to {os.path.join(export_dir, 'summary.json')}")
        
        return results
