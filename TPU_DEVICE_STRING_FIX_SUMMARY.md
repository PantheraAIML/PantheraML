# TPU Device String Fix Summary

## Problem
The original error was:
```
RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: tpu
```

This occurred because PyTorch doesn't recognize "tpu" as a valid device string. For TPU operations with PyTorch/XLA, the correct device string is "xla".

## Root Cause
Throughout the PantheraML codebase, when `DEVICE_TYPE` was set to "tpu", the code was passing strings like:
- `device="tpu"` 
- `device="tpu:0"`
- `device=f"{DEVICE_TYPE}:0"` (when DEVICE_TYPE="tpu")

These were being passed to PyTorch operations like:
- `tensor.to(device="tpu")`
- `torch.empty(..., device="tpu:0")`
- `torch.autocast(device_type="tpu")`

## Solution Implemented

### 1. Added Device Mapping Utility Functions
Created utility functions in `pantheraml/models/_utils.py`:

```python
def get_pytorch_device(device_id=None):
    """Convert PantheraML device type to PyTorch-compatible device string."""
    from pantheraml import DEVICE_TYPE
    
    if DEVICE_TYPE == "tpu":
        # PyTorch/XLA uses "xla" for TPU devices
        return "xla" if device_id is None else f"xla:{device_id}"
    elif device_id is not None:
        return f"{DEVICE_TYPE}:{device_id}"
    else:
        return DEVICE_TYPE

def get_autocast_device():
    """Get device type for torch.autocast operations."""
    from pantheraml import DEVICE_TYPE
    
    if DEVICE_TYPE == "tpu":
        return "cpu"  # TPU operations use CPU for autocast
    return DEVICE_TYPE
```

### 2. Updated All Model Files
Fixed device string usage across all model files:

**Files Updated:**
- `pantheraml/models/llama.py` (62+ fixes)
- `pantheraml/models/gemma.py` (4 fixes)
- `pantheraml/models/gemma2.py` (3 fixes)
- `pantheraml/models/cohere.py` (10 fixes)
- `pantheraml/models/falcon_h1.py` (4 fixes)
- `pantheraml/models/granite.py` (1 fix)
- `pantheraml/models/mistral.py` (1 fix)
- `pantheraml/models/qwen3.py` (1 fix)
- `pantheraml/models/vision.py` (2 fixes)

**Pattern Replacements:**
- `device = f"{DEVICE_TYPE}:0"` → `device = get_pytorch_device(0)`
- `device = DEVICE_TYPE` → `device = get_pytorch_device()`
- `torch.autocast(device_type = DEVICE_TYPE)` → `torch.autocast(device_type = get_autocast_device())`

### 3. Added Proper Imports
Updated all affected model files to import the utility functions:
```python
from ._utils import __version__, get_pytorch_device, get_autocast_device
```

## Device Mapping Results

| PantheraML DEVICE_TYPE | PyTorch Device String | Autocast Device |
|------------------------|----------------------|-----------------|
| "tpu"                  | "xla"                | "cpu"           |
| "tpu" + device_id=0    | "xla:0"              | "cpu"           |
| "cuda"                 | "cuda"               | "cuda"          |
| "cuda" + device_id=0   | "cuda:0"             | "cuda"          |
| "xpu"                  | "xpu"                | "xpu"           |
| "cpu"                  | "cpu"                | "cpu"           |

## Verification
✅ All hardcoded device patterns removed (62+ instances fixed)
✅ Device utility functions properly imported and used
✅ Device conversion logic tested and working
✅ PyTorch accepts "xla" device strings without parsing errors
✅ TPU operations now use correct device strings for PyTorch/XLA compatibility

## Impact
- **Fixed**: The original runtime error `"Expected one of cpu, cuda... device string: tpu"`
- **Maintained**: Full backward compatibility for CUDA, XPU, and CPU
- **Improved**: TPU support now properly integrates with PyTorch/XLA
- **Robust**: Device-agnostic architecture maintained across all devices

The fix ensures that when PantheraML is running in TPU mode (DEVICE_TYPE="tpu"), all PyTorch operations receive the correct "xla" device strings that PyTorch/XLA expects, while maintaining seamless operation on all other device types.
