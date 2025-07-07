# PantheraML Complete API Reference

## Table of Contents
1. [Core Model Functions](#core-model-functions)
2. [Training & Fine-tuning](#training--fine-tuning)
3. [Saving & Loading](#saving--loading)
4. [Model Format Conversions](#model-format-conversions)
5. [Benchmarking & Evaluation](#benchmarking--evaluation)
6. [Distributed Training](#distributed-training)
7. [TPU Support](#tpu-support)
8. [Kernel Operations](#kernel-operations)
9. [Utility Functions](#utility-functions)
10. [CLI Commands](#cli-commands)
11. [Configuration Classes](#configuration-classes)
12. [Exception Classes](#exception-classes)

---

## Core Model Functions

### FastLanguageModel

#### `FastLanguageModel.from_pretrained()`
```python
FastLanguageModel.from_pretrained(
    model_name: str,
    max_seq_length: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = True,
    token: Optional[str] = None,
    device_map: Optional[Union[str, Dict]] = None,
    rope_scaling: Optional[Dict] = None,
    fix_tokenizer: bool = True,
    trust_remote_code: bool = False,
    use_gradient_checkpointing: Optional[str] = "unsloth",
    resize_model_vocab: Optional[int] = None,
    revision: Optional[str] = None,
    *args, **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]
```
**Description**: Load a pretrained language model optimized for PantheraML.

**Parameters**:
- `model_name`: HuggingFace model identifier or local path
- `max_seq_length`: Maximum sequence length for the model
- `dtype`: Data type (torch.float16, torch.bfloat16, etc.)
- `load_in_4bit`: Enable 4-bit quantization for memory efficiency
- `token`: HuggingFace authentication token
- `device_map`: Device mapping for multi-GPU setups
- `rope_scaling`: RoPE scaling configuration for extended context
- `fix_tokenizer`: Apply PantheraML tokenizer fixes
- `trust_remote_code`: Allow execution of custom model code
- `use_gradient_checkpointing`: Gradient checkpointing method
- `resize_model_vocab`: Resize vocabulary to specific size
- `revision`: Model revision/branch to load

**Returns**: Tuple of (model, tokenizer)

**Example**:
```python
from pantheraml import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

#### `FastLanguageModel.get_peft_model()`
```python
FastLanguageModel.get_peft_model(
    model: PreTrainedModel,
    r: int = 16,
    target_modules: Optional[List[str]] = None,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    random_state: int = 3407,
    use_gradient_checkpointing: Optional[str] = "unsloth",
    use_rslora: bool = False,
    loftq_config: Optional[Dict] = None,
    **kwargs
) -> PeftModel
```
**Description**: Convert a model to use Parameter Efficient Fine-Tuning (PEFT) with LoRA.

**Parameters**:
- `model`: Base model to apply PEFT to
- `r`: LoRA rank (higher = more parameters)
- `target_modules`: List of module names to apply LoRA to
- `lora_alpha`: LoRA scaling parameter
- `lora_dropout`: Dropout rate for LoRA layers
- `bias`: Bias handling ("none", "all", "lora_only")
- `task_type`: Task type for PEFT
- `random_state`: Random seed for reproducibility
- `use_gradient_checkpointing`: Gradient checkpointing method
- `use_rslora`: Use Rank-Stabilized LoRA
- `loftq_config`: LoftQ quantization configuration

**Returns**: PEFT-enabled model

**Example**:
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
```

#### `FastLanguageModel.for_inference()`
```python
FastLanguageModel.for_inference(model: PreTrainedModel) -> PreTrainedModel
```
**Description**: Optimize model for inference by disabling training-specific features.

**Parameters**:
- `model`: Model to optimize for inference

**Returns**: Inference-optimized model

**Example**:
```python
FastLanguageModel.for_inference(model)
```

#### `FastLanguageModel.for_training()`
```python
FastLanguageModel.for_training(model: PreTrainedModel) -> PreTrainedModel
```
**Description**: Re-enable training mode for a model.

**Parameters**:
- `model`: Model to enable training for

**Returns**: Training-enabled model

### Model Loading Utilities

#### `get_model_name()`
```python
get_model_name(
    model_name: str,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    bnb_4bit_quant_storage: torch.dtype = torch.uint8,
) -> str
```
**Description**: Get the appropriate model name with quantization suffixes.

#### `check_model_config()`
```python
check_model_config(model_name: str) -> Dict[str, Any]
```
**Description**: Check and validate model configuration.

---

## Training & Fine-tuning

### Training Configuration

#### `PatchDPOTrainer()`
```python
PatchDPOTrainer() -> None
```
**Description**: Patch DPO trainer for compatibility with PantheraML.

**Example**:
```python
from pantheraml import PatchDPOTrainer
PatchDPOTrainer()  # Call before using DPOTrainer
```

#### `is_bfloat16_supported()`
```python
is_bfloat16_supported() -> bool
```
**Description**: Check if bfloat16 is supported on current hardware.

#### `get_chat_template()`
```python
get_chat_template(
    tokenizer: PreTrainedTokenizer,
    chat_template: Optional[str] = None,
    mapping: Optional[Dict] = None,
) -> str
```
**Description**: Get or set chat template for instruction-following models.

### Training Utilities

#### `apply_chat_template()`
```python
apply_chat_template(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
    **kwargs
) -> str
```
**Description**: Apply chat template to conversation messages.

#### `standardize_sharegpt()`
```python
standardize_sharegpt(
    conversations: List[Dict],
    mapping: Optional[Dict[str, str]] = None,
) -> List[Dict]
```
**Description**: Standardize ShareGPT format conversations.

---

## Saving & Loading

### Core Saving Functions

#### `pantheraml_save_model()`
```python
pantheraml_save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    save_directory: Union[str, os.PathLike],
    save_method: str = "lora",
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with PantheraML",
    private: Optional[bool] = None,
    create_pr: bool = False,
    revision: str = None,
    commit_description: str = "Upload model trained with PantheraML 2x faster",
    tags: List[str] = None,
    temporary_location: str = "_pantheraml_temporary_saved_buffers",
    maximum_memory_usage: float = 0.9,
) -> Tuple[str, str]
```
**Description**: Main function for saving PantheraML models in various formats.

**Parameters**:
- `model`: Model to save
- `tokenizer`: Associated tokenizer
- `save_directory`: Directory to save to
- `save_method`: "lora", "merged_16bit", "merged_4bit", or "merged_4bit_forced"
- `push_to_hub`: Whether to push to HuggingFace Hub
- `token`: HuggingFace authentication token
- `max_shard_size`: Maximum size per model shard
- `safe_serialization`: Use safetensors format
- `commit_message`: Git commit message for Hub uploads
- `tags`: Model tags for Hub
- `temporary_location`: Temporary directory for processing
- `maximum_memory_usage`: Maximum memory usage ratio

**Returns**: Tuple of (save_directory, username)

#### `pantheraml_save_pretrained_merged()`
```python
pantheraml_save_pretrained_merged(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    save_method: str = "merged_16bit",
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    tags: List[str] = None,
    temporary_location: str = "_pantheraml_temporary_saved_buffers",
    maximum_memory_usage: float = 0.75,
) -> None
```
**Description**: Save model with merged LoRA weights.

#### `pantheraml_push_to_hub_merged()`
```python
pantheraml_push_to_hub_merged(
    self,
    repo_id: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    save_method: str = "merged_16bit",
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with PantheraML",
    private: Optional[bool] = None,
    token: Union[bool, str, None] = None,
    max_shard_size: Union[int, str, None] = "5GB",
    create_pr: bool = False,
    safe_serialization: bool = True,
    revision: str = None,
    commit_description: str = "Upload model trained with PantheraML 2x faster",
    tags: Optional[List[str]] = None,
    temporary_location: str = "_pantheraml_temporary_saved_buffers",
    maximum_memory_usage: float = 0.75,
) -> None
```
**Description**: Push merged model directly to HuggingFace Hub.

### Repository Management

#### `create_huggingface_repo()`
```python
create_huggingface_repo(
    model: PreTrainedModel,
    save_directory: str,
    token: Optional[str] = None,
    private: bool = False,
) -> Tuple[str, HfApi]
```
**Description**: Create a new repository on HuggingFace Hub.

#### `upload_to_huggingface()`
```python
upload_to_huggingface(
    model: PreTrainedModel,
    save_directory: str,
    token: str,
    method: str,
    extra: str = "",
    file_location: Optional[str] = None,
    old_username: Optional[str] = None,
    private: Optional[bool] = None,
    create_config: bool = True,
) -> str
```
**Description**: Upload files to HuggingFace Hub with proper metadata.

---

## Model Format Conversions

### GGUF Conversion

#### `pantheraml_save_pretrained_gguf()`
```python
pantheraml_save_pretrained_gguf(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    quantization_method: str = "fast_quantized",
    first_conversion: Optional[str] = None,
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    private: Optional[bool] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    tags: List[str] = None,
    temporary_location: str = "_pantheraml_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
) -> None
```
**Description**: Convert and save model to GGUF format for llama.cpp compatibility.

**Parameters**:
- `quantization_method`: Quantization type ("q4_k_m", "q8_0", "q5_k_m", etc.)
- `first_conversion`: Initial conversion format ("f16", "bf16", "f32", "q8_0")

#### `pantheraml_push_to_hub_gguf()`
```python
pantheraml_push_to_hub_gguf(
    self,
    repo_id: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    quantization_method: str = "fast_quantized",
    first_conversion: Optional[str] = None,
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with PantheraML",
    private: Optional[bool] = None,
    token: Union[bool, str, None] = None,
    max_shard_size: Union[int, str, None] = "5GB",
    create_pr: bool = False,
    safe_serialization: bool = True,
    revision: str = None,
    commit_description: str = "Upload model trained with PantheraML 2x faster",
    tags: Optional[List[str]] = None,
    temporary_location: str = "_pantheraml_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
) -> None
```
**Description**: Convert and push GGUF model to HuggingFace Hub.

#### `save_to_gguf()`
```python
save_to_gguf(
    model_type: str,
    model_dtype: str,
    is_sentencepiece: bool = False,
    model_directory: str = "pantheraml_finetuned_model",
    quantization_method: Union[str, List[str]] = "fast_quantized",
    first_conversion: Optional[str] = None,
    _run_installer = None,
) -> Tuple[List[str], bool]
```
**Description**: Core GGUF conversion function.

#### `print_quantization_methods()`
```python
print_quantization_methods() -> None
```
**Description**: Print all available quantization methods with descriptions.

**Example**:
```python
from pantheraml.save import print_quantization_methods
print_quantization_methods()
```

### GGML Conversion

#### `pantheraml_convert_lora_to_ggml_and_save_locally()`
```python
pantheraml_convert_lora_to_ggml_and_save_locally(
    self,
    save_directory: str,
    tokenizer: PreTrainedTokenizer,
    temporary_location: str = "_pantheraml_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
) -> None
```
**Description**: Convert LoRA adapters to GGML format and save locally.

#### `pantheraml_convert_lora_to_ggml_and_push_to_hub()`
```python
pantheraml_convert_lora_to_ggml_and_push_to_hub(
    self,
    tokenizer: PreTrainedTokenizer,
    repo_id: str,
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Converted LoRA to GGML with PantheraML",
    private: Optional[bool] = None,
    token: Union[bool, str, None] = None,
    create_pr: bool = False,
    revision: str = None,
    commit_description: str = "Convert LoRA to GGML format using PantheraML",
    temporary_location: str = "_pantheraml_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
) -> None
```
**Description**: Convert LoRA to GGML and push to HuggingFace Hub.

### Ollama Integration

#### `create_ollama_modelfile()`
```python
create_ollama_modelfile(
    tokenizer: PreTrainedTokenizer,
    gguf_location: str
) -> Optional[str]
```
**Description**: Create Ollama Modelfile for GGUF models.

#### `push_to_ollama()`
```python
push_to_ollama(
    tokenizer: PreTrainedTokenizer,
    gguf_location: str,
    username: str,
    model_name: str,
    tag: str
) -> None
```
**Description**: Push model to Ollama registry.

#### `create_ollama_model()`
```python
create_ollama_model(
    username: str,
    model_name: str,
    tag: str,
    modelfile_path: str
) -> None
```
**Description**: Create Ollama model from Modelfile.

#### `push_to_ollama_hub()`
```python
push_to_ollama_hub(
    username: str,
    model_name: str,
    tag: str
) -> None
```
**Description**: Push created model to Ollama Hub.

---

## Benchmarking & Evaluation

### Direct Benchmark Functions

#### `benchmark_mmlu()`
```python
benchmark_mmlu(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str = "auto",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    subjects: Optional[List[str]] = None,
    export: bool = False,
    export_path: Optional[str] = None,
    **kwargs
) -> BenchmarkResult
```
**Description**: Run MMLU (Massive Multitask Language Understanding) benchmark.

**Parameters**:
- `model`: Model to evaluate
- `tokenizer`: Associated tokenizer
- `device`: Device to run on ("cuda", "cpu", "auto")
- `batch_size`: Evaluation batch size
- `max_samples`: Limit number of samples (for testing)
- `subjects`: Specific MMLU subjects to evaluate
- `export`: Whether to export results to file
- `export_path`: Custom export path

**Returns**: BenchmarkResult object

**Example**:
```python
from pantheraml import benchmark_mmlu

result = benchmark_mmlu(model, tokenizer, export=True)
print(f"MMLU Accuracy: {result.accuracy:.2%}")
```

#### `benchmark_hellaswag()`
```python
benchmark_hellaswag(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str = "auto",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    export: bool = False,
    export_path: Optional[str] = None,
    **kwargs
) -> BenchmarkResult
```
**Description**: Run HellaSwag commonsense reasoning benchmark.

#### `benchmark_arc()`
```python
benchmark_arc(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    challenge: bool = False,
    device: str = "auto",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    export: bool = False,
    export_path: Optional[str] = None,
    **kwargs
) -> BenchmarkResult
```
**Description**: Run ARC (AI2 Reasoning Challenge) benchmark.

**Parameters**:
- `challenge`: Use ARC-Challenge (harder) instead of ARC-Easy

#### `run_benchmark()`
```python
run_benchmark(
    model_name: str,
    benchmark: str,
    device: str = "auto",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    export: bool = False,
    **kwargs
) -> BenchmarkResult
```
**Description**: Run any benchmark by name.

**Parameters**:
- `benchmark`: Benchmark name ("mmlu", "hellaswag", "arc_easy", "arc_challenge")

### Benchmark Classes

#### `BenchmarkRunner`
```python
class BenchmarkRunner:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        load_in_4bit: bool = True,
        torch_dtype: Optional[torch.dtype] = None
    )
    
    def run(
        self,
        benchmark: str,
        batch_size: int = 1,
        max_samples: Optional[int] = None,
        **kwargs
    ) -> BenchmarkResult
    
    def run_all(
        self,
        batch_size: int = 1,
        max_samples: Optional[int] = None,
        **kwargs
    ) -> Dict[str, BenchmarkResult]
    
    def export_results(
        self,
        filename: str,
        format: str = "json"
    ) -> None
```

**Example**:
```python
from pantheraml import BenchmarkRunner

runner = BenchmarkRunner("microsoft/DialoGPT-medium")
mmlu_result = runner.run("mmlu", batch_size=4)
all_results = runner.run_all()
runner.export_results("results.json")
```

#### `PantheraBench`
```python
class PantheraBench:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "auto"
    )
    
    def mmlu(
        self,
        batch_size: int = 1,
        max_samples: Optional[int] = None,
        subjects: Optional[List[str]] = None,
        export: bool = False,
        **kwargs
    ) -> BenchmarkResult
    
    def hellaswag(
        self,
        batch_size: int = 1,
        max_samples: Optional[int] = None,
        export: bool = False,
        **kwargs
    ) -> BenchmarkResult
    
    def arc(
        self,
        challenge: bool = False,
        batch_size: int = 1,
        max_samples: Optional[int] = None,
        export: bool = False,
        **kwargs
    ) -> BenchmarkResult
```

**Example**:
```python
from pantheraml import PantheraBench

bench = PantheraBench(model, tokenizer)
result = bench.mmlu(export=True)
```

### Result Classes

#### `BenchmarkResult`
```python
@dataclass
class BenchmarkResult:
    benchmark_name: str
    accuracy: float
    total_questions: int
    correct_answers: int
    execution_time: float
    device_info: Dict[str, Any]
    timestamp: str
    model_name: str
    subjects: Optional[List[str]] = None
    per_subject_results: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]
    def to_json(self) -> str
    def export(self, filename: str, format: str = "json") -> None
```

---

## Distributed Training

### Setup Functions

#### `setup_distributed_training()`
```python
setup_distributed_training() -> None
```
**Description**: Initialize distributed training environment.

#### `DistributedTrainingManager`
```python
class DistributedTrainingManager:
    def __init__(self)
    
    def setup(self) -> None
    def cleanup(self) -> None
    def prepare_model(self, model: PreTrainedModel) -> PreTrainedModel
    def is_main_process(self) -> bool
    
    @property
    def world_size(self) -> int
    
    @property
    def local_rank(self) -> int
    
    @property
    def global_rank(self) -> int
```

**Example**:
```python
from pantheraml.distributed import DistributedTrainingManager

dist_manager = DistributedTrainingManager()
dist_manager.setup()

model = dist_manager.prepare_model(model)
# ... training code ...

if dist_manager.is_main_process():
    model.save_pretrained("final_model")

dist_manager.cleanup()
```

---

## TPU Support

### TPU Functions

#### `setup_tpu_training()`
```python
setup_tpu_training() -> None
```
**Description**: Initialize TPU training environment.

#### `TPUMonitor`
```python
class TPUMonitor:
    def __init__(self)
    
    def get_metrics(self) -> Dict[str, Any]
    def log_step(self, step: int, metrics: Dict[str, Any]) -> None
    def start_monitoring(self) -> None
    def stop_monitoring(self) -> None
```

**Example**:
```python
from pantheraml.tpu import setup_tpu_training, TPUMonitor

setup_tpu_training()
monitor = TPUMonitor()

# During training
metrics = monitor.get_metrics()
print(f"TPU Utilization: {metrics['utilization']:.1%}")
```

---

## Kernel Operations

### Optimization Functions

#### `optimize_memory_usage()`
```python
optimize_memory_usage(
    model: PreTrainedModel,
    max_memory_usage: float = 0.85,
    enable_gradient_checkpointing: bool = True,
    use_fast_kernels: bool = True,
) -> None
```
**Description**: Optimize model memory usage.

#### `fast_dequantize()`
```python
fast_dequantize(
    weight: torch.Tensor,
    quant_state: Any,
    quant_type: str
) -> torch.Tensor
```
**Description**: Fast dequantization for 4-bit weights.

#### `get_lora_parameters_bias()`
```python
get_lora_parameters_bias(
    model: PreTrainedModel,
    bias: str = "none"
) -> List[torch.nn.Parameter]
```
**Description**: Get LoRA parameters with bias handling.

---

## Utility Functions

### General Utilities

#### `show_install_info()`
```python
show_install_info() -> None
```
**Description**: Display PantheraML installation information.

#### `is_pantheraml_available()`
```python
is_pantheraml_available() -> bool
```
**Description**: Check if PantheraML is properly installed.

#### `get_pantheraml_version()`
```python
get_pantheraml_version() -> str
```
**Description**: Get PantheraML version string.

### Tokenizer Utilities

#### `fix_tokenizer_bos_token()`
```python
fix_tokenizer_bos_token(
    tokenizer: PreTrainedTokenizer
) -> Tuple[bool, Optional[str]]
```
**Description**: Fix tokenizer BOS token issues for chat templates.

#### `fix_sentencepiece_gguf()`
```python
fix_sentencepiece_gguf(model_directory: str) -> None
```
**Description**: Fix SentencePiece tokenizer for GGUF conversion.

### File Utilities

#### `check_if_sentencepiece_model()`
```python
check_if_sentencepiece_model(
    model: PreTrainedModel,
    temporary_location: str = "_pantheraml_sentencepiece_temp"
) -> bool
```
**Description**: Check if model uses SentencePiece tokenizer.

---

## CLI Commands

### Main CLI Script: `pantheraml-cli.py`

#### Training Command
```bash
pantheraml-cli.py train [OPTIONS]
```

**Options**:
- `--model_name`: Model to fine-tune
- `--dataset`: Training dataset path
- `--output_dir`: Output directory
- `--max_seq_length`: Maximum sequence length
- `--load_in_4bit`: Enable 4-bit quantization
- `--learning_rate`: Learning rate
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Gradient accumulation steps

#### Benchmark Command
```bash
pantheraml-cli.py benchmark [OPTIONS]
```

**Options**:
- `--benchmark`: Enable benchmarking mode
- `--benchmark_type`: Benchmark type (mmlu, hellaswag, arc, all)
- `--model_name`: Model to benchmark
- `--device`: Device to use
- `--batch_size`: Evaluation batch size
- `--max_samples`: Limit samples for testing
- `--export`: Export results to file

#### Convert Command
```bash
pantheraml-cli.py convert [OPTIONS]
```

**Options**:
- `--input`: Input model path
- `--output`: Output path
- `--format`: Output format (gguf, ggml)
- `--quantization_method`: Quantization method
- `--push_to_hub`: Push to HuggingFace Hub

#### Example Commands
```bash
# Basic training
pantheraml-cli.py --model_name unsloth/llama-2-7b-chat --dataset data.json

# Benchmarking
pantheraml-cli.py --benchmark --benchmark_type mmlu --model_name my_model --export

# GGUF conversion
pantheraml-cli.py --model_name my_model --save_method gguf --quantization_method q4_k_m

# Multi-GPU training
pantheraml-cli.py --model_name unsloth/llama-2-13b --dataset large_data.json --multi_gpu

# Push to Hub
pantheraml-cli.py --model_name my_model --push_to_hub --token $HF_TOKEN --repo_id username/model_name
```

---

## Configuration Classes

### TrainingConfig
```python
@dataclass
class TrainingConfig:
    model_name: str
    dataset_path: str
    output_dir: str = "outputs"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    logging_steps: int = 1
    save_strategy: str = "epoch"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig"
    
    def to_dict(self) -> Dict[str, Any]
```

### BenchmarkConfig
```python
@dataclass
class BenchmarkConfig:
    model_name: str
    benchmark_type: str
    device: str = "auto"
    batch_size: int = 1
    max_samples: Optional[int] = None
    export: bool = False
    export_path: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BenchmarkConfig"
    
    def to_dict(self) -> Dict[str, Any]
```

---

## Exception Classes

### PantheraMLError
```python
class PantheraMLError(Exception):
    """Base exception for PantheraML"""
    pass
```

### ModelLoadingError
```python
class ModelLoadingError(PantheraMLError):
    """Error during model loading"""
    pass
```

### TrainingError
```python
class TrainingError(PantheraMLError):
    """Error during training"""
    pass
```

### BenchmarkError
```python
class BenchmarkError(PantheraMLError):
    """Error during benchmarking"""
    pass
```

### ConversionError
```python
class ConversionError(PantheraMLError):
    """Error during model format conversion"""
    pass
```

---

## Import Summary

### Main Imports
```python
# Core functionality
from pantheraml import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

# Benchmarking
from pantheraml import (
    benchmark_mmlu, benchmark_hellaswag, benchmark_arc,
    run_benchmark, BenchmarkRunner, PantheraBench
)

# Distributed training
from pantheraml.distributed import setup_distributed_training, DistributedTrainingManager

# TPU support
from pantheraml.tpu import setup_tpu_training, TPUMonitor

# Saving functions
from pantheraml.save import (
    pantheraml_save_model, print_quantization_methods,
    create_huggingface_repo, push_to_ollama
)

# Utilities
from pantheraml import (
    show_install_info, is_pantheraml_available, get_pantheraml_version
)

# Kernel operations
from pantheraml.kernels.utils import optimize_memory_usage

# Chat utilities
from pantheraml import get_chat_template, apply_chat_template, standardize_sharegpt
```

### All Available Imports
```python
# Complete import list
from pantheraml import (
    # Core model functions
    FastLanguageModel,
    
    # Training utilities
    PatchDPOTrainer,
    is_bfloat16_supported,
    get_chat_template,
    apply_chat_template,
    standardize_sharegpt,
    
    # Benchmarking functions
    benchmark_mmlu,
    benchmark_hellaswag,
    benchmark_arc,
    run_benchmark,
    BenchmarkRunner,
    PantheraBench,
    BenchmarkResult,
    
    # Utility functions
    show_install_info,
    is_pantheraml_available,
    get_pantheraml_version,
    
    # Configuration classes
    TrainingConfig,
    BenchmarkConfig,
    
    # Exception classes
    PantheraMLError,
    ModelLoadingError,
    TrainingError,
    BenchmarkError,
    ConversionError,
)
```

---

This comprehensive API reference covers every function, class, and method available in PantheraML. Each entry includes the complete function signature, parameter descriptions, return types, and usage examples. This documentation is designed to be used for creating complete documentation pages for the PantheraML library.
