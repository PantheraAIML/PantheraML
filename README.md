<div align="center">

# 🦥 PantheraML

**Enhanced LLM Fine-tuning with Multi-GPU Support**

*Built on the excellent foundation of Unsloth*

### Finetune Gemma 3n, Qwen3, Llama 4, Phi-4 & Mistral 2x faster with 80% less VRAM!
### 🚀 Now with Multi-GPU Support!

![](https://i.ibb.co/sJ7RhGG/image-41.png)

</div>

## 🙏 Credits & Acknowledgments

**PantheraML** is built upon the excellent work of the [Unsloth team](https://github.com/unslothai/unsloth). We extend our heartfelt gratitude to Daniel Han-Chen and the entire Unsloth community for creating such an outstanding foundation for efficient LLM fine-tuning.

### What PantheraML Adds:
- 🔥 **Multi-GPU distributed training** - Scale across multiple GPUs seamlessly
- 🚀 **Enhanced memory optimization** - Better memory management for large models  
- 📊 **Advanced monitoring** - Real-time GPU memory and training metrics
- 🔧 **Extended compatibility** - Support for more model architectures

*Original Unsloth: https://github.com/unslothai/unsloth*


## ⚡ Quickstart

> **🎯 New!** PantheraML now supports multi-GPU training and experimental TPU support!

- **Install with pip (recommended)** for Linux devices:
```bash
pip install pantheraml
```

- **Quick Start Example:**
```python
import torch
from pantheraml import FastLanguageModel

# All original Unsloth functionality, enhanced with multi-GPU support!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/DialoGPT-medium",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)
```
