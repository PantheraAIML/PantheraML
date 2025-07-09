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

def get_tpu_safe_device(ordinal=None):
    """
    Get TPU device safely, handling driver initialization issues.
    
    Args:
        ordinal: Optional TPU ordinal (device ID)
        
    Returns:
        torch.device: Safe device object
    """
    import torch
    
    if DEVICE_TYPE != "tpu":
        return torch.device(get_pytorch_device(ordinal))
    
    try:
        import torch_xla.core.xla_model as xm
        
        # Check if TPU is available first
        world_size = xm.xrt_world_size()
        if world_size == 0:
            raise RuntimeError("No TPU devices available")
            
        # Get device with proper error handling
        if ordinal is not None:
            device = xm.xla_device(ordinal=ordinal)
        else:
            device = xm.xla_device()
            
        return device
        
    except Exception as e:
        # Fallback to CPU if TPU fails
        print(f"TPU initialization failed: {e}")
        print("Falling back to CPU device")
        return torch.device("cpu")

def initialize_tpu_context():
    """
    Initialize TPU context properly to prevent driver errors.
    
    Returns:
        torch.device or None: Initialized TPU device or None if failed
    """
    if DEVICE_TYPE != "tpu":
        return None
        
    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        
        # Force XLA initialization with proper sequencing
        device = xm.xla_device()
        
        # Create a dummy tensor to initialize the device context
        dummy = torch.tensor([1.0], device=device)
        xm.mark_step()  # Synchronize XLA operations
        
        # Clean up dummy tensor
        del dummy
        
        return device
        
    except Exception as e:
        print(f"TPU context initialization failed: {e}")
        return None

def safe_device_placement(tensor_or_model, target_device):
    """
    Safely place tensor or model on target device with proper XLA handling.
    
    Args:
        tensor_or_model: PyTorch tensor or model
        target_device: Target device
        
    Returns:
        Tensor or model placed on target device
    """
    import torch
    
    try:
        if hasattr(target_device, 'type') and target_device.type == "xla":
            # For XLA devices, ensure proper synchronization
            import torch_xla.core.xla_model as xm
            result = tensor_or_model.to(target_device)
            xm.mark_step()  # Synchronize XLA operations
            return result
        else:
            return tensor_or_model.to(target_device)
            
    except Exception as e:
        print(f"Device placement failed: {e}")
        # Fallback to CPU
        return tensor_or_model.to("cpu")

def get_device_with_fallback(device_id=None):
    """
    Get device with automatic fallback for TPU driver issues.
    
    Args:
        device_id: Optional device ID
        
    Returns:
        torch.device: Safe device object
    """
    import torch
    
    if DEVICE_TYPE == "tpu":
        # Try TPU initialization with fallback
        tpu_device = get_tpu_safe_device(device_id)
        if tpu_device.type == "cpu":
            print("TPU not available, using CPU")
        return tpu_device
    else:
        return torch.device(get_pytorch_device(device_id))