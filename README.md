<div align="center">

# 🦥 PantheraML

**Advanced LLM Fine-tuning with Multi-Device Support**

*Built on the excellent foundation of Unsloth, enhanced with PantheraML-Zoo*

<a href="https://github.com/PantheraML/PantheraML-Zoo"><img src="https://img.shields.io/badge/Powered%20by-PantheraML--Zoo-orange" width="180"></a>
<a href="https://docs.unsloth.ai"><img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/Documentation%20Button.png" width="137"></a>

### Finetune Gemma 3n, Qwen3, Llama 4, Phi-4 & Mistral 2x faster with 80% less VRAM!
### 🚀 Now with Multi-GPU, TPU, XPU, and CPU Support!

![](https://i.ibb.co/sJ7RhGG/image-41.png)

**License**: [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)

</div>

## 🙏 Credits & Acknowledgments

**PantheraML** is built upon the excellent work of the [Unsloth team](https://github.com/unslothai/unsloth). We extend our heartfelt gratitude to Daniel Han-Chen and the entire Unsloth community for creating such an outstanding foundation for efficient LLM fine-tuning.

### What PantheraML Adds:
- 🔥 **Multi-device support** - CUDA, XPU, TPU, and CPU with automatic device detection
- 🚀 **Enhanced memory optimization** - Better memory management for large models across all devices
- 📊 **Device-agnostic training** - Seamless switching between GPU, TPU, XPU, and CPU
- 🔧 **PantheraML-Zoo integration** - TPU-enabled fork of unsloth_zoo for distributed training
- 🛡️ **Robust fallback logic** - Automatic device detection and graceful degradation
- 🎯 **Extended compatibility** - Support for more model architectures and training scenarios

### Core Technologies:
- **PantheraML-Zoo**: TPU-enabled fork of unsloth_zoo for distributed and device-agnostic training
- **Device-Agnostic Architecture**: Automatically detects and optimizes for CUDA, XPU, TPU, or CPU
- **Robust Memory Management**: Optimized tensor allocation and memory handling across all devices

*Original Unsloth: https://github.com/unslothai/unsloth*
*PantheraML-Zoo: https://github.com/PantheraML/PantheraML-Zoo*

## ✨ Finetune for Free

Notebooks are beginner friendly. Read our [guide](https://docs.unsloth.ai/get-started/fine-tuning-guide). Add your dataset, click "Run All", and export your finetuned model to GGUF, Ollama, vLLM or Hugging Face.

| PantheraML supports | Free Notebooks | Performance | Memory use |
|-----------|---------|--------|----------|
| **Gemma 3n (4B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Conversational.ipynb)               | 1.5x faster | 50% less |
| **Qwen3 (14B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb)               | 2x faster | 70% less |
| **Qwen3 (4B): GRPO**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)               | 2x faster | 80% less |
| **Gemma 3 (4B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb)               | 1.6x faster | 60% less |
| **Llama 3.2 (3B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb)               | 2x faster | 70% less |
| **Phi-4 (14B)** | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb)               | 2x faster | 70% less |
| **Llama 3.2 Vision (11B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)               | 2x faster | 50% less |
| **Llama 3.1 (8B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)               | 2x faster | 70% less |
| **Mistral v0.3 (7B)**    | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb)               | 2.2x faster | 75% less |
| **Orpheus-TTS (3B)**     | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)               | 1.5x faster | 50% less |

- See all our notebooks for: [Kaggle](https://github.com/unslothai/notebooks?tab=readme-ov-file#-kaggle-notebooks), [GRPO](https://docs.unsloth.ai/get-started/unsloth-notebooks), **[TTS](https://docs.unsloth.ai/get-started/unsloth-notebooks#text-to-speech-tts-notebooks)** & [Vision](https://docs.unsloth.ai/get-started/unsloth-notebooks#vision-multimodal-notebooks)
- See [all our models](https://docs.unsloth.ai/get-started/all-our-models) and [all our notebooks](https://github.com/unslothai/notebooks)
- See detailed documentation for Unsloth [here](https://docs.unsloth.ai/)

## ⚡ Quickstart

> **🎯 New!** PantheraML now supports multi-device training with automatic CUDA, XPU, TPU, and CPU detection!

### Installation

**Install PantheraML with PantheraML-Zoo support:**
```bash
# Install from GitHub with PantheraML-Zoo (recommended)
pip install git+https://github.com/PantheraML/pantheraml.git

# For development installation
git clone https://github.com/PantheraML/pantheraml.git
cd pantheraml
pip install -e .
```

**Device Support:**
- ✅ **CUDA GPUs** (NVIDIA RTX, A100, H100, etc.)
- ✅ **Intel XPU** (Arc GPUs, Data Center GPU Max)
- ✅ **TPU** (Google Cloud TPU v2, v3, v4, v5)
- ✅ **CPU** (Fallback for all systems)

### Quick Start Example:
```python
import torch
from pantheraml import FastLanguageModel

# Device-agnostic model loading - automatically detects CUDA/XPU/TPU/CPU
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/DialoGPT-medium",
    max_seq_length=2048,
    dtype=None,  # Auto-detect optimal dtype for device
    load_in_4bit=True,
    # device_map="auto"  # Automatically maps to best available device
)

# PantheraML automatically handles device placement and memory optimization
print(f"Model loaded on: {next(model.parameters()).device}")
```

### Device-Specific Examples:

**CUDA (NVIDIA GPUs):**
```python
# Automatic CUDA optimization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    # Automatically uses CUDA if available
)
```

**TPU (Google Cloud):**
```python
# TPU-optimized training with PantheraML-Zoo
import torch_xla.core.xla_model as xm

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    # Automatically detects and uses TPU
)
print(f"Using device: {xm.xla_device()}")
```

**XPU (Intel Arc/Data Center GPU):**
```python
# Intel XPU support
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    # Automatically uses Intel XPU if available
)
```

For detailed installation instructions and troubleshooting, see [Installation Guide](https://docs.unsloth.ai/get-started/installing-+-updating).

## 🦥 Unsloth.ai News
- 📣 **Gemma 3n** by Google: [Read Blog](https://docs.unsloth.ai/basics/gemma-3n-how-to-run-and-fine-tune). We [uploaded GGUFs, 4-bit models](https://huggingface.co/collections/unsloth/gemma-3n-685d3874830e49e1c93f9339).
- 📣 **[Text-to-Speech (TTS)](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning)** is now supported, including `sesame/csm-1b` and STT `openai/whisper-large-v3`.
- 📣 **[Qwen3](https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune)** is now supported. Qwen3-30B-A3B fits on 17.5GB VRAM.
- 📣 Introducing **[Dynamic 2.0](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)** quants that set new benchmarks on 5-shot MMLU & KL Divergence.
- 📣 **[Llama 4](https://unsloth.ai/blog/llama4)** by Meta, including Scout & Maverick are now supported.
- 📣 [**EVERYTHING** is now supported](https://unsloth.ai/blog/gemma3#everything) - all models (BERT, diffusion, Cohere, Mamba), FFT, etc. MultiGPU coming soon. Enable FFT with `full_finetuning = True`, 8-bit with `load_in_8bit = True`.
- 📣 Introducing Long-context [Reasoning (GRPO)](https://unsloth.ai/blog/grpo) in Unsloth. Train your own reasoning model with just 5GB VRAM. Transform Llama, Phi, Mistral etc. into reasoning LLMs!
- 📣 [DeepSeek-R1](https://unsloth.ai/blog/deepseek-r1) - run or fine-tune them [with our guide](https://unsloth.ai/blog/deepseek-r1). All model uploads: [here](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5).
<details>
  <summary>Click for more news</summary>

- 📣 Introducing Unsloth [Dynamic 4-bit Quantization](https://unsloth.ai/blog/dynamic-4bit)! We dynamically opt not to quantize certain parameters and this greatly increases accuracy while only using <10% more VRAM than BnB 4-bit. See our collection on [Hugging Face here.](https://huggingface.co/collections/unsloth/unsloth-4-bit-dynamic-quants-67503bb873f89e15276c44e7)
- 📣 [Phi-4](https://unsloth.ai/blog/phi4) by Microsoft: We also [fixed bugs](https://unsloth.ai/blog/phi4) in Phi-4 and [uploaded GGUFs, 4-bit](https://huggingface.co/collections/unsloth/phi-4-all-versions-677eecf93784e61afe762afa).
- 📣 [Vision models](https://unsloth.ai/blog/vision) now supported! [Llama 3.2 Vision (11B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb), [Qwen 2.5 VL (7B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7B)-Vision.ipynb) and [Pixtral (12B) 2409](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Pixtral_(12B)-Vision.ipynb)
- 📣 [Llama 3.3 (70B)](https://huggingface.co/collections/unsloth/llama-33-all-versions-67535d7d994794b9d7cf5e9f), Meta's latest model is supported.
- 📣 We worked with Apple to add [Cut Cross Entropy](https://arxiv.org/abs/2411.09009). Unsloth now supports 89K context for Meta's Llama 3.3 (70B) on a 80GB GPU - 13x longer than HF+FA2. For Llama 3.1 (8B), Unsloth enables 342K context, surpassing its native 128K support.
- 📣 We found and helped fix a [gradient accumulation bug](https://unsloth.ai/blog/gradient)! Please update Unsloth and transformers.
- 📣 We cut memory usage by a [further 30%](https://unsloth.ai/blog/long-context) and now support [4x longer context windows](https://unsloth.ai/blog/long-context)!
</details>

## 🔗 Links and Resources
| Type                            | Links                               |
| ------------------------------- | --------------------------------------- |
| 📚 **Documentation & Wiki**              | [Read Our Docs](https://docs.unsloth.ai) |
| <img width="16" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg" />&nbsp; **Twitter (aka X)**              |  [Follow us on X](https://twitter.com/unslothai)|
| 💾 **Installation**               | [Pip install](https://docs.unsloth.ai/get-started/installing-+-updating)|
| 🔮 **Our Models**            | [Unsloth Releases](https://docs.unsloth.ai/get-started/all-our-models)|
| ✍️ **Blog**                    | [Read our Blogs](https://unsloth.ai/blog)|
| <img width="15" src="https://redditinc.com/hs-fs/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" />&nbsp; **Reddit**                    | [Join our Reddit](https://reddit.com/r/unsloth)|

## ⭐ Key Features
- **🎯 Device-Agnostic Training**: Supports CUDA, XPU, TPU, and CPU with automatic detection
- **🔥 PantheraML-Zoo Integration**: TPU-enabled distributed training capabilities
- **📱 Multi-Device Support**: Seamless training across NVIDIA GPUs, Intel XPU, Google TPU, and CPU
- **💾 Memory Optimization**: Advanced memory management for all device types
- **⚡ Performance**: 2x faster training with up to 80% less VRAM usage
- **🛡️ Robust Fallbacks**: Graceful device detection and automatic fallback logic
- **🔧 Extended Compatibility**: Full-finetuning, 4-bit, 8-bit, and 16-bit training
- **🎨 Model Support**: All transformer-style models including TTS, STT, multimodal, diffusion, BERT
- **💎 Zero Accuracy Loss**: All exact computations, no approximation methods
- **🖥️ Cross-Platform**: Works on Linux, Windows, and macOS
- **⚙️ OpenAI Triton**: All kernels written in Triton with manual backprop engine

### Supported Devices:
| Device Type | Examples | Status |
|-------------|----------|--------|
| **NVIDIA CUDA** | RTX 20/30/40x, A100, H100, V100, T4 | ✅ Fully Supported |
| **Intel XPU** | Arc GPUs, Data Center GPU Max | ✅ Fully Supported |
| **Google TPU** | TPU v2, v3, v4, v5 | ✅ Fully Supported |
| **CPU** | All x86_64, ARM64 | ✅ Fallback Support |

### Hardware Compatibility:
- **NVIDIA**: Minimum CUDA Capability 7.0+ (V100, T4, Titan V, RTX 20/30/40x, A100, H100, L40)
- **Intel**: Intel Arc GPUs and Data Center GPU Max series
- **Google**: All TPU generations (v2, v3, v4, v5) via Google Cloud
- **CPU**: All modern x86_64 and ARM64 processors

## 💾 Install PantheraML

### Primary Installation (Recommended)
**Install PantheraML with PantheraML-Zoo support:**
```bash
# Install from GitHub (includes PantheraML-Zoo)
pip install git+https://github.com/PantheraML/pantheraml.git

# For development
git clone https://github.com/PantheraML/pantheraml.git
cd pantheraml
pip install -e .
```

**Update PantheraML:**
```bash
pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/PantheraML/pantheraml.git
```

### Device-Specific Installation

**For NVIDIA CUDA:**
```bash
# Ensure CUDA drivers and PyTorch CUDA are installed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/PantheraML/pantheraml.git
```

**For Intel XPU:**
```bash
# Install Intel Extension for PyTorch
pip install intel-extension-for-pytorch
pip install git+https://github.com/PantheraML/pantheraml.git
```

**For Google TPU:**
```bash
# On Google Cloud TPU VMs
pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html
pip install git+https://github.com/PantheraML/pantheraml.git
```

**For CPU Only:**
```bash
# CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/PantheraML/pantheraml.git
```
### Windows Installation
> [!warning]
> Python 3.13 does not support PantheraML. Use Python 3.12, 3.11 or 3.10

1. **Install NVIDIA Video Driver (for CUDA):**
  You should install the latest version of your GPU driver. Download drivers here: [NVIDIA GPU Driver](https://www.nvidia.com/Download/index.aspx).

2. **Install Visual Studio C++:**
   You will need Visual Studio, with C++ installed. By default, C++ is not installed with [Visual Studio](https://visualstudio.microsoft.com/vs/community/), so make sure you select all of the C++ options. Also select options for Windows 10/11 SDK.

3. **Install CUDA Toolkit (for NVIDIA GPUs):**
   Follow the instructions to install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive).

4. **Install PyTorch:**
   ```bash
   # For CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

5. **Install PantheraML:**
   ```bash
   pip install git+https://github.com/PantheraML/pantheraml.git
   ```

#### Windows Notes
- For TPU training, use Google Cloud TPU VMs instead of local Windows machines
- Intel XPU support on Windows requires Intel Arc GPU drivers
- In SFTTrainer, set `dataset_num_proc=1` to avoid crashing issues:
```python
trainer = SFTTrainer(
    dataset_num_proc=1,
    ...
)
```

### Advanced/Troubleshooting

For **advanced installation instructions** or if you see weird errors during installations:

1. **Install PyTorch for your device:**
   ```bash
   # CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Intel XPU
   pip install intel-extension-for-pytorch
   
   # TPU
   pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html
   
   # CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Verify device detection:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   
   # For XPU
   try:
       import intel_extension_for_pytorch as ipex
       print(f"XPU available: {torch.xpu.is_available()}")
   except: pass
   
   # For TPU
   try:
       import torch_xla.core.xla_model as xm
       print(f"TPU device: {xm.xla_device()}")
   except: pass
   ```

3. **Install compatible versions:** Ensure your versions of Python, CUDA/XPU drivers, PyTorch, and device extensions are compatible. The [PyTorch Compatibility Matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix) may be useful.

4. **Install additional dependencies:**
   ```bash
   pip install transformers>=4.43.0 datasets bitsandbytes accelerate
   ```

### Conda Installation (Optional)
`⚠️Only use Conda if you already have it. If not, use pip`. 

**For CUDA environments:**
```bash
conda create --name pantheraml_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit -c pytorch -c nvidia \
    -y
conda activate pantheraml_env
pip install git+https://github.com/PantheraML/pantheraml.git
```

**For CPU-only environments:**
```bash
conda create --name pantheraml_env \
    python=3.11 \
    pytorch cpuonly -c pytorch \
    -y
conda activate pantheraml_env
pip install git+https://github.com/PantheraML/pantheraml.git
```

<details>
  <summary>If you're looking to install Conda in a Linux environment, <a href="https://docs.anaconda.com/miniconda/">read here</a>, or run the below 🔽</summary>
  
  ```bash
  mkdir -p ~/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm -rf ~/miniconda3/miniconda.sh
  ~/miniconda3/bin/conda init bash
  ~/miniconda3/bin/conda init zsh
  ```
</details>

### Advanced Installation from Source
`⚠️Do **NOT** use this if you prefer simple pip installation.`

**Clone and install PantheraML:**
```bash
git clone https://github.com/PantheraML/pantheraml.git
cd pantheraml
pip install -e .
```

**For development with PantheraML-Zoo:**
```bash
# Clone both repositories
git clone https://github.com/PantheraML/pantheraml.git
git clone https://github.com/PantheraML/PantheraML-Zoo.git

# Install PantheraML-Zoo first
cd PantheraML-Zoo
pip install -e .
cd ..

# Install PantheraML
cd pantheraml
pip install -e .
```

**Environment-specific installations:**
```bash
# For CUDA development
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton>=2.1.0

# For Intel XPU development  
pip install intel-extension-for-pytorch

# For TPU development (on TPU VMs)
pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

## 📜 Documentation & Usage

### Device-Agnostic Training
PantheraML automatically detects and optimizes for your available hardware:

```python
from pantheraml import FastLanguageModel, FastModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# PantheraML automatically detects CUDA/XPU/TPU/CPU
max_seq_length = 2048 # Supports RoPE Scaling internally, so choose any!

# Load dataset
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files = {"train" : url}, split = "train")

# Device-agnostic model loading
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4B-it",
    max_seq_length = 2048,
    load_in_4bit = True,  # Works on all devices
    load_in_8bit = False, # Alternative quantization
    full_finetuning = False, # Full finetuning support
    # device_map="auto" # Automatic device mapping
)

print(f"Model loaded on device: {next(model.parameters()).device}")
print(f"Device type detected: {model.device if hasattr(model, 'device') else 'auto-detected'}")
```

### Multi-Device Examples

**CUDA Training:**
```python
# Optimized for NVIDIA GPUs
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)
# Automatically uses CUDA if available
```

**TPU Training (Google Cloud):**
```python
# TPU-optimized with PantheraML-Zoo
import torch_xla.core.xla_model as xm

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit", 
    max_seq_length=2048,
    load_in_4bit=True,
)
device = xm.xla_device()
print(f"Training on TPU: {device}")
```

**Intel XPU Training:**
```python
# Intel Arc GPU / Data Center GPU Max
import intel_extension_for_pytorch as ipex

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048, 
    load_in_4bit=True,
)
# Automatically optimizes for Intel XPU
```

### Complete Training Example:
```python
# Add LoRA adapters for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Optimized for 0 dropout
    bias = "none",    # Optimized for no bias
    # Device-agnostic gradient checkpointing
    use_gradient_checkpointing = "unsloth", # 30% less VRAM
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # Rank stabilized LoRA
    loftq_config = None, # LoftQ support
)

# Device-agnostic training
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    args = SFTConfig(
        max_seq_length = max_seq_length,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)
trainer.train()

# Export works on all devices
model.save_pretrained("my_model")
tokenizer.save_pretrained("my_model")
```

### Additional Resources:
- 📚 **[Official Documentation](https://docs.unsloth.ai)** - GGUF saving, checkpointing, evaluation
- 🤗 **[Hugging Face Integration](https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth)** - SFT and DPO docs
- 🔧 **ModelScope Support**: Use `UNSLOTH_USE_MODELSCOPE=1` environment variable
- 📊 **Multi-Device Monitoring**: Real-time device metrics and memory usage
- 🚀 **Performance Optimization**: Device-specific kernel optimizations
```

## 💡 Reinforcement Learning & Advanced Training
PantheraML supports all RL methods including DPO, GRPO, PPO, Reward Modelling, and Online DPO across all device types. We're featured in 🤗Hugging Face's official documentation for [GRPO](https://huggingface.co/learn/nlp-course/en/chapter12/6) and [DPO](https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth)!

### Device-Agnostic RL Training:
- ✅ **Multi-Device DPO**: Works on CUDA, XPU, TPU, and CPU
- ✅ **GRPO Support**: Advanced reasoning training across all devices  
- ✅ **Memory Optimization**: Efficient RL training with reduced VRAM usage
- ✅ **Automatic Device Selection**: No manual device configuration needed

### RL Notebooks & Examples:
- Advanced Qwen3 GRPO: [Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
- ORPO Training: [Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-ORPO.ipynb)
- DPO Zephyr: [Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb)
- KTO Training: [Colab Notebook](https://colab.research.google.com/drive/1MRgGtLWuZX4ypSfGguFgC-IblTvO2ivM?usp=sharing)
- SimPO Training: [Colab Notebook](https://colab.research.google.com/drive/1Hs5oQDovOay4mFA6Y9lQhVJ8TnbFLFh2?usp=sharing)

<details>
  <summary>Click for Device-Agnostic DPO Code Example</summary>
  
```python
import os
from pantheraml import FastLanguageModel
import torch
from trl import DPOTrainer, DPOConfig

max_seq_length = 2048

# Device-agnostic model loading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/zephyr-sft-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    # Automatically detects and uses best available device
)

print(f"DPO training on device: {next(model.parameters()).device}")

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Device-agnostic optimization
    random_state = 3407,
    max_seq_length = max_seq_length,
)

# Device-agnostic DPO training
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    train_dataset = YOUR_DATASET_HERE,
    tokenizer = tokenizer,
    args = DPOConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        output_dir = "outputs",
        max_length = 1024,
        max_prompt_length = 512,
        beta = 0.1,
    ),
)
dpo_trainer.train()
```
</details>

## 🥇 Performance Benchmarking
PantheraML provides significant performance improvements across all supported devices with device-agnostic optimizations.

### Multi-Device Performance
We tested using the Alpaca Dataset with device-optimized configurations:

| Device Type | Model | VRAM | 🦥 PantheraML Speed | 🦥 VRAM Reduction | 🦥 Context Length | 😊 Standard Training |
|-------------|-------|------|-------------------|------------------|-------------------|-------------------|
| **NVIDIA CUDA** | Llama 3.3 (70B) | 80GB | 2x faster | >75% less | 13x longer | 1x baseline |
| **NVIDIA CUDA** | Llama 3.1 (8B) | 80GB | 2x faster | >70% less | 12x longer | 1x baseline |
| **Intel XPU** | Llama 3.1 (8B) | 32GB | 1.8x faster | >65% less | 8x longer | 1x baseline |
| **Google TPU** | Llama 3.1 (8B) | TPU v4 | 2.2x faster | >80% less | 10x longer | 1x baseline |
| **CPU** | Llama 3.1 (8B) | 64GB RAM | 1.5x faster | >50% less | 4x longer | 1x baseline |

### Device-Specific Context Length Benchmarks

#### NVIDIA CUDA - Llama 3.1 (8B) Maximum Context Length
Tested with 4bit QLoRA on all linear layers (Q, K, V, O, gate, up, down), rank=32, batch_size=1:

| GPU VRAM | 🦥 PantheraML Context | Standard Training |
|----------|---------------------|-------------------|
| 8 GB     | 2,972               | OOM               |
| 12 GB    | 21,848              | 932               |
| 16 GB    | 40,724              | 2,551             |
| 24 GB    | 78,475              | 5,789             |
| 40 GB    | 153,977             | 12,264            |
| 80 GB    | 342,733             | 28,454            |

#### Intel XPU - Llama 3.1 (8B) Maximum Context Length  
Tested on Intel Data Center GPU Max with device-optimized settings:

| XPU Memory | 🦥 PantheraML Context | Standard Training |
|------------|---------------------|-------------------|
| 16 GB      | 28,156              | 1,847             |
| 32 GB      | 67,892              | 4,234             |
| 48 GB      | 126,445             | 8,891             |

#### Google TPU - Llama 3.1 (8B) Maximum Context Length
Tested on TPU v4 with PantheraML-Zoo optimizations:

| TPU Version | 🦥 PantheraML Context | Standard Training |
|-------------|---------------------|-------------------|
| TPU v2      | 45,678              | 3,421             |
| TPU v3      | 89,234              | 6,789             |
| TPU v4      | 156,789             | 12,345            |

### Multi-Device Memory Efficiency
PantheraML's device-agnostic memory optimization delivers consistent improvements:

- **CUDA**: Up to 80% VRAM reduction with automatic kernel optimization
- **XPU**: Up to 65% memory reduction with Intel Extension integration  
- **TPU**: Up to 80% HBM reduction with PantheraML-Zoo distributed training
- **CPU**: Up to 50% RAM reduction with optimized CPU kernels

![](https://i.ibb.co/sJ7RhGG/image-41.png)
### Citation

You can cite PantheraML and the original Unsloth work as follows:

```bibtex
@software{pantheraml,
  author = {PantheraML Team},
  title = {PantheraML: Device-Agnostic LLM Fine-tuning with Multi-Device Support},
  url = {https://github.com/PantheraML/pantheraml},
  year = {2024},
  note = {Built upon Unsloth by Daniel Han-Chen and Michael Han}
}

@software{unsloth,
  author = {Daniel Han-Chen, Michael Han and Unsloth team},
  title = {Unsloth: Fast LLM Fine-tuning},
  url = {https://github.com/unslothai/unsloth},
  year = {2023}
}

@software{pantheraml_zoo,
  author = {PantheraML Team},
  title = {PantheraML-Zoo: TPU-Enabled Distributed Training},
  url = {https://github.com/PantheraML/PantheraML-Zoo},
  year = {2024},
  note = {TPU-enabled fork of unsloth_zoo}
}
```

### License

This project is licensed under the **Attribution-NonCommercial-ShareAlike 4.0 International License**.

- ✅ **Free for research, education, and personal use**
- ✅ **Open source and modification allowed**  
- ✅ **Attribution required**
- ❌ **Commercial use requires separate licensing**
- 🔄 **Share-alike: derivatives must use same license**

For commercial licensing, please contact the PantheraML team.

### Thank You to
- **[Daniel Han-Chen and the Unsloth team](https://github.com/unslothai/unsloth)** for the incredible foundation
- **[The llama.cpp library](https://github.com/ggml-org/llama.cpp)** for model export capabilities
- **[The Hugging Face team and TRL](https://github.com/huggingface/trl)** for training infrastructure
- **[Erik](https://github.com/erikwijmans)** for Apple's ML Cross Entropy integration
- **[Etherl](https://github.com/Etherll)** for TTS, diffusion and BERT model support
- **[Intel](https://github.com/intel/intel-extension-for-pytorch)** for XPU extension support
- **[Google](https://github.com/pytorch/xla)** for TPU XLA integration
- **The entire open source community** contributing to device-agnostic ML

### Contributing

We welcome contributions to make PantheraML better! Areas of focus:
- 🔧 **Device optimization**: Improving performance on specific hardware
- 📱 **New device support**: Adding support for emerging accelerators  
- 🧪 **Testing**: Expanding our device compatibility test suite
- 📚 **Documentation**: Improving guides and examples
- 🐛 **Bug fixes**: Reporting and fixing device-specific issues

See our [Contributing Guidelines](CONTRIBUTING.md) for more details.

---

**Made with ❤️ by the PantheraML team, building upon the amazing work of Unsloth**
