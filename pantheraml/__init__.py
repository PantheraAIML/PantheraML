#
# PantheraML is built upon the excellent work of the Unsloth team.
# Original Unsloth: https://github.com/unslothai/unsloth
# We gratefully acknowledge their contributions to efficient LLM fine-tuning.
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

import warnings, importlib, sys
from packaging.version import Version
import os, re, subprocess, inspect
import numpy as np

# Check if modules that need patching are already imported
critical_modules = ['trl', 'transformers', 'peft']
already_imported = [mod for mod in critical_modules if mod in sys.modules]

# This check is critical because PantheraML optimizes these libraries by modifying
# their code at import time. If they're imported first, the original (slower, 
# more memory-intensive) implementations will be used instead of PantheraML's
# optimized versions, potentially causing OOM errors or slower training.

if already_imported:
    # stacklevel=2 makes warning point to user's import line rather than this library code,
    # showing them exactly where to fix the import order in their script
    warnings.warn(
        f"WARNING: PantheraML should be imported before {', '.join(already_imported)} "
        f"to ensure all optimizations are applied. Your code may run slower or encounter "
        f"memory issues without these optimizations.\n\n"
        f"Please restructure your imports with 'import pantheraml' at the top of your file.",
        stacklevel = 2,
    )
pass

# PantheraML now supports multi-GPU setups! This includes:
# - Distributed Data Parallel (DDP) training
# - Model parallelism across multiple GPUs  
# - Automatic device mapping and memory optimization
# - Mixed precision training across GPUs
# Use the PantheraMLDistributedTrainer for easy multi-GPU training!

# Fixes https://github.com/unslothai/unsloth/issues/1266
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# [TODO] Check why some GPUs don't work
#    "pinned_use_cuda_host_register:True,"\
#    "pinned_num_register_threads:8"

# Hugging Face Hub faster downloads
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
pass

# XET is slower in Colab - investigate why
keynames = "\n" + "\n".join(os.environ.keys())
if "HF_XET_HIGH_PERFORMANCE" not in os.environ:
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
pass
# Disable XET cache sine it eats too much space
if "HF_XET_CHUNK_CACHE_SIZE_BYTES" not in os.environ:
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
pass
if "\nCOLAB_" in keynames:
    os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "0"
pass

# Log PantheraML is being used
os.environ["UNSLOTH_IS_PRESENT"] = "1"

try:
    import torch
except ModuleNotFoundError:
    raise ImportError(
        "PantheraML: Pytorch is not installed. Go to https://pytorch.org/.\n"\
        "We have some installation instructions on our Github page."
    )
except Exception as exception:
    raise exception
pass

def get_device_type():
    # Allow overriding device type for testing/development
    override_device = os.environ.get("PANTHERAML_DEVICE_TYPE", "").lower()
    if override_device in ["cuda", "xpu", "tpu", "cpu"]:
        if override_device == "tpu":
            print("⚠️  EXPERIMENTAL: TPU support forced via PANTHERAML_DEVICE_TYPE. This is experimental and may have limitations.")
        elif override_device == "cpu":
            print("⚠️  WARNING: CPU device forced via PANTHERAML_DEVICE_TYPE")
        return override_device
    
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    # Experimental TPU support
    try:
        import torch_xla.core.xla_model as xm
        if xm.xla_device():
            print("⚠️  EXPERIMENTAL: TPU support detected. This is experimental and may have limitations.")
            return "tpu"
    except ImportError:
        pass
    
    # Check if we're in development/CLI mode
    if os.environ.get("PANTHERAML_DEV_MODE", "0") == "1":
        print("⚠️  WARNING: Running in development mode on unsupported device")
        print("🚫 PantheraML requires NVIDIA GPUs, Intel GPUs, or TPUs for training")
        return "cpu"  # Return cpu for development mode
    
    raise NotImplementedError("PantheraML currently only works on NVIDIA GPUs, Intel GPUs, and TPUs (experimental).")
pass
DEVICE_TYPE : str = get_device_type()

# Reduce VRAM usage by reducing fragmentation
# And optimize pinning of memory
if DEVICE_TYPE == "cuda" and os.environ.get("UNSLOTH_VLLM_STANDBY", "0")=="0":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
        "expandable_segments:True,"\
        "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
elif DEVICE_TYPE == "tpu":
    # TPU-specific optimizations with Phase 1 enhancements
    print("🧪 EXPERIMENTAL: Applying TPU-specific optimizations with Phase 1 enhancements...")
    
    # Basic TPU environment variables
    os.environ["XLA_USE_BF16"] = "1"  # Enable bfloat16 for TPUs
    os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"  # Limit tensor allocator
    
    # Phase 1: Enhanced TPU initialization
    try:
        from .kernels.tpu_kernels import initialize_tpu_kernels, get_tpu_status
        from .distributed import get_tpu_status as dist_get_tpu_status
        
        print("🧪 TPU: Initializing Phase 1 kernel enhancements...")
        
        # Initialize TPU kernels with enhanced error handling
        if initialize_tpu_kernels():
            # Get comprehensive TPU status
            tpu_status = get_tpu_status()
            if not tpu_status.get("error"):
                print(f"✅ TPU: Phase 1 kernels initialized successfully")
                print(f"🧪 TPU: Available memory: {tpu_status.get('memory_info', {}).get('gb_limit', 'Unknown')} GB")
            else:
                print(f"⚠️ TPU: Kernel status check failed: {tpu_status.get('error')}")
        else:
            print("⚠️ TPU: Phase 1 kernel initialization failed, using basic TPU support")
            
    except ImportError as e:
        print(f"⚠️ TPU: Phase 1 enhancements not available: {e}")
        print("🧪 TPU: Falling back to basic TPU support")
    except Exception as e:
        print(f"⚠️ TPU: Phase 1 initialization error: {e}")
        print("🧪 TPU: Continuing with basic TPU support")

# We support Pytorch 2
# Fixes https://github.com/unslothai/unsloth/issues/38
torch_version = str(torch.__version__).split(".")
major_torch, minor_torch = torch_version[0], torch_version[1]
major_torch, minor_torch = int(major_torch), int(minor_torch)
if (major_torch < 2):
    raise ImportError("PantheraML only supports Pytorch 2 for now. Please update your Pytorch to 2.1.\n"\
                      "We have some installation instructions on our Github page.")
elif (major_torch == 2) and (minor_torch < 2):
    # Disable expandable_segments
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]
pass

# Fix Xformers performance issues since 0.0.25
import importlib.util
from pathlib import Path
from importlib.metadata import version as importlib_version
from packaging.version import Version
try:
    xformers_version = importlib_version("xformers")
    if Version(xformers_version) < Version("0.0.29"):
        xformers_location = importlib.util.find_spec("xformers").origin
        xformers_location = os.path.split(xformers_location)[0]
        cutlass = Path(xformers_location) / "ops" / "fmha" / "cutlass.py"

        if cutlass.exists():
            with open(cutlass, "r+") as f:
                text = f.read()
                # See https://github.com/facebookresearch/xformers/issues/1176#issuecomment-2545829591
                if "num_splits_key=-1," in text:
                    text = text.replace("num_splits_key=-1,", "num_splits_key=None,")
                    f.seek(0)
                    f.write(text)
                    f.truncate()
                    print("PantheraML: Patching Xformers to fix some performance issues.")
                pass
            pass
        pass
    pass
except:
    pass
pass

# Torch 2.4 has including_emulation
if DEVICE_TYPE == "cuda":
    major_version, minor_version = torch.cuda.get_device_capability()
    SUPPORTS_BFLOAT16 = (major_version >= 8)

    old_is_bf16_supported = torch.cuda.is_bf16_supported
    if "including_emulation" in str(inspect.signature(old_is_bf16_supported)):
        def is_bf16_supported(including_emulation = False):
            return old_is_bf16_supported(including_emulation)
        torch.cuda.is_bf16_supported = is_bf16_supported
    else:
        def is_bf16_supported(): return SUPPORTS_BFLOAT16
        torch.cuda.is_bf16_supported = is_bf16_supported
    pass
elif DEVICE_TYPE == "xpu":
    # torch.xpu.is_bf16_supported() does not have including_emulation
    # set SUPPORTS_BFLOAT16 as torch.xpu.is_bf16_supported()
    SUPPORTS_BFLOAT16 = torch.xpu.is_bf16_supported()
pass


# For Gradio HF Spaces?
# if "SPACE_AUTHOR_NAME" not in os.environ and "SPACE_REPO_NAME" not in os.environ:
try:
    import triton
    HAS_TRITON = True
except ImportError:
    triton = None
    HAS_TRITON = False

if DEVICE_TYPE == "cuda" and HAS_TRITON:
    libcuda_dirs = lambda: None
    if Version(triton.__version__) >= Version("3.0.0"):
        try: from triton.backends.nvidia.driver import libcuda_dirs
        except: pass
    else: from triton.common.build import libcuda_dirs

    # Try loading bitsandbytes and triton
    import bitsandbytes as bnb
    try:
        cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
        libcuda_dirs()
    except:
        warnings.warn(
            "PantheraML: Running `ldconfig /usr/lib64-nvidia` to link CUDA."\
        )

        if os.path.exists("/usr/lib64-nvidia"):
            os.system("ldconfig /usr/lib64-nvidia")
        elif os.path.exists("/usr/local"):
            # Sometimes bitsandbytes cannot be linked properly in Runpod for example
            possible_cudas = subprocess.check_output(["ls", "-al", "/usr/local"]).decode("utf-8").split("\n")
            find_cuda = re.compile(r"[\s](cuda\-[\d\.]{2,})$")
            possible_cudas = [find_cuda.search(x) for x in possible_cudas]
            possible_cudas = [x.group(1) for x in possible_cudas if x is not None]

            # Try linking cuda folder, or everything in local
            if len(possible_cudas) == 0:
                os.system("ldconfig /usr/local/")
            else:
                find_number = re.compile(r"([\d\.]{2,})")
                latest_cuda = np.argsort([float(find_number.search(x).group(1)) for x in possible_cudas])[::-1][0]
                latest_cuda = possible_cudas[latest_cuda]
                os.system(f"ldconfig /usr/local/{latest_cuda}")
        pass

        importlib.reload(bnb)
        importlib.reload(triton)
        try:
            libcuda_dirs = lambda: None
            if Version(triton.__version__) >= Version("3.0.0"):
                try: from triton.backends.nvidia.driver import libcuda_dirs
                except: pass
            else: from triton.common.build import libcuda_dirs
            cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
            libcuda_dirs()
        except:
            warnings.warn(
                "Unsloth: CUDA is not linked properly.\n"\
                "Try running `python -m bitsandbytes` then `python -m xformers.info`\n"\
                "We tried running `ldconfig /usr/lib64-nvidia` ourselves, but it didn't work.\n"\
                "You need to run in your terminal `sudo ldconfig /usr/lib64-nvidia` yourself, then import Unsloth.\n"\
                "Also try `sudo ldconfig /usr/local/cuda-xx.x` - find the latest cuda version.\n"\
                "Unsloth will still run for now, but maybe it might crash - let's hope it works!"
            )
    pass
elif DEVICE_TYPE == "xpu":
    # currently intel xpu will not support bnb, will add support in the future
    # TODO: check triton for intel installed properly.
    pass

# Check for PantheraML Zoo (TPU-enabled fork of unsloth_zoo)
try:
    # Try PantheraML Zoo first (our TPU-enabled fork)
    try:
        pantheraml_zoo_version = importlib_version("pantheraml_zoo")
        import pantheraml_zoo as unsloth_zoo
        print("✅ PantheraML Zoo loaded (TPU support enabled)")
    except ImportError:
        # Fallback to original unsloth_zoo with warning
        try:
            unsloth_zoo_version = importlib_version("unsloth_zoo")
            import unsloth_zoo
            print("⚠️ Using original unsloth_zoo (limited TPU support)")
            print("   For full TPU support, install: pip install git+https://github.com/PantheraAIML/PantheraML-Zoo.git")
        except ImportError:
            raise ImportError(
                "PantheraML requires PantheraML Zoo for optimal performance.\n"
                "Install with: pip install git+https://github.com/PantheraAIML/PantheraML-Zoo.git\n"
                "Or fallback: pip install unsloth_zoo"
            )
except Exception as e:
    print(f"Warning: Zoo import failed: {e}")
    print("Some features may be limited without PantheraML Zoo")
pass

from .models import *
from .models import __version__
from .save import *
from .chat_templates import *
from .tokenizer_utils import *
from .trainer import *

# Patch TRL trainers for backwards compatibility
def _patch_trl_trainer():
    """Patch TRL trainer for backwards compatibility."""
    try:
        import trl
        # Apply any necessary patches here
        pass
    except ImportError:
        # TRL not available, skip patching
        pass

_patch_trl_trainer()

# Experimental TPU support
from .distributed import (
    is_tpu_available,
    setup_multi_tpu,
    cleanup_multi_tpu,
    MultiTPUConfig,
    get_tpu_rank,
    get_tpu_world_size,
    is_tpu_main_process,
)

# Built-in Benchmarking Support
from .benchmarks import (
    benchmark_mmlu,
    benchmark_hellaswag,
    benchmark_arc,
    PantheraBench,
    MMLUBenchmark,
    HellaSwagBenchmark,
    ARCBenchmark,
    BenchmarkResult,
)

# Add TPU-related and benchmarking exports
__all__ = [
    # ... existing exports ...
    # Experimental TPU support
    "is_tpu_available",
    "setup_multi_tpu", 
    "cleanup_multi_tpu",
    "MultiTPUConfig",
    "PantheraMLTPUTrainer",
    # Built-in benchmarking
    "benchmark_mmlu",
    "benchmark_hellaswag", 
    "benchmark_arc",
    "PantheraBench",
    "MMLUBenchmark",
    "HellaSwagBenchmark",
    "ARCBenchmark",
    "BenchmarkResult",
]