# Copyright 2024 PantheraML
# Licensed under the Attribution-NonCommercial-ShareAlike 4.0 International License.
# See LICENSE file for details.
"""
Utility functions for device-agnostic model operations.
"""
import os

# Get device type from environment
DEVICE_TYPE = os.environ.get("UNSLOTH_DEVICE_TYPE", "cuda")

def get_pytorch_device(device_id=None):
    """
    Convert PantheraML device type to PyTorch-compatible device string.
    
    Args:
        device_id: Optional device ID (e.g., 0 for first device)
        
    Returns:
        str: PyTorch-compatible device string
    """
    if DEVICE_TYPE == "tpu":
        # PyTorch/XLA uses "xla" for TPU devices
        return "xla" if device_id is None else f"xla:{device_id}"
    elif device_id is not None:
        return f"{DEVICE_TYPE}:{device_id}"
    else:
        return DEVICE_TYPE

def get_autocast_device():
    """
    Get device type for torch.autocast operations.
    
    Returns:
        str: Device type compatible with torch.autocast
    """
    if DEVICE_TYPE == "tpu":
        return "cpu"  # TPU operations use CPU for autocast
    return DEVICE_TYPE