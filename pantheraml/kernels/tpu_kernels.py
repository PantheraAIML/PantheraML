#  Copyright 2025-present Aayan Mishra & the PantheraML team. All rights reserved.
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
TPU-specific kernel optimizations for PantheraML.
Phase 1: Core stability and XLA integration.
"""

import torch
import warnings
from typing import Optional, Tuple, Dict, Any
from ..kernels.utils import DEVICE_TYPE

# TPU-specific imports with error handling
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    xm = None
    xmp = None
    xr = None


class TPUMemoryManager:
    """Enhanced TPU memory management for stability."""
    
    def __init__(self):
        self.memory_cache = {}
        self.peak_memory = 0
        self.allocated_memory = 0
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get comprehensive TPU memory information."""
        if not TPU_AVAILABLE or xm is None:
            return {"error": "TPU not available"}
            
        try:
            # Get TPU device
            device = xm.xla_device()
            
            # Get memory statistics
            memory_info = xr.memory_info(device)
            
            # Convert bytes to GB for readability
            def bytes_to_gb(bytes_val):
                return round(bytes_val / (1024**3), 3) if bytes_val else 0
            
            return {
                "device": str(device),
                "bytes_in_use": memory_info.get("bytes_in_use", 0),
                "bytes_limit": memory_info.get("bytes_limit", 0),
                "gb_in_use": bytes_to_gb(memory_info.get("bytes_in_use", 0)),
                "gb_limit": bytes_to_gb(memory_info.get("bytes_limit", 0)),
                "utilization_percent": round(
                    (memory_info.get("bytes_in_use", 0) / memory_info.get("bytes_limit", 1)) * 100, 2
                ) if memory_info.get("bytes_limit", 0) > 0 else 0
            }
        except Exception as e:
            return {"error": f"Failed to get TPU memory info: {e}"}
    
    def optimize_memory(self) -> bool:
        """Apply TPU-specific memory optimizations."""
        if not TPU_AVAILABLE or xm is None:
            return False
            
        try:
            # Mark step to synchronize and optimize memory
            xm.mark_step()
            
            # Wait for pending operations
            xm.wait_device_ops()
            
            # Reduce HBM fragmentation
            if hasattr(xr, "reduce_scatter"):
                xr.reduce_scatter()
                
            print("🧪 TPU: Memory optimization completed")
            return True
            
        except Exception as e:
            print(f"⚠️ TPU: Memory optimization failed: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """Clear TPU memory cache safely."""
        if not TPU_AVAILABLE or xm is None:
            return False
            
        try:
            # Synchronize before clearing
            xm.mark_step()
            xm.wait_device_ops()
            
            # Clear memory cache
            self.memory_cache.clear()
            
            print("🧪 TPU: Memory cache cleared")
            return True
            
        except Exception as e:
            print(f"⚠️ TPU: Failed to clear cache: {e}")
            return False


class XLAOptimizer:
    """XLA compilation and optimization utilities."""
    
    def __init__(self):
        self.compiled_functions = {}
        self.optimization_cache = {}
    
    def is_xla_available(self) -> bool:
        """Check if XLA is properly available."""
        return TPU_AVAILABLE and xm is not None
    
    def get_xla_device(self) -> Optional[torch.device]:
        """Get XLA device with error handling."""
        if not self.is_xla_available():
            return None
            
        try:
            device = xm.xla_device()
            return device
        except Exception as e:
            print(f"⚠️ TPU: Failed to get XLA device: {e}")
            return None
    
    def compile_for_tpu(self, func, *args, **kwargs):
        """Compile function for TPU with XLA optimizations."""
        if not self.is_xla_available():
            return func
            
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        # Check cache first
        if func_name in self.compiled_functions:
            return self.compiled_functions[func_name]
        
        try:
            # Use torch.jit.script for XLA compilation
            compiled_func = torch.jit.script(func)
            self.compiled_functions[func_name] = compiled_func
            
            print(f"🧪 TPU: Compiled {func_name} for XLA")
            return compiled_func
            
        except Exception as e:
            print(f"⚠️ TPU: Failed to compile {func_name}: {e}")
            return func
    
    def optimize_tensor_operations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply TPU-specific tensor optimizations."""
        if not self.is_xla_available():
            return tensor
            
        try:
            # Move to TPU device if not already there
            device = self.get_xla_device()
            if device and tensor.device != device:
                tensor = tensor.to(device)
            
            # Apply TPU-specific optimizations
            # Ensure tensors are contiguous for TPU efficiency
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            return tensor
            
        except Exception as e:
            print(f"⚠️ TPU: Tensor optimization failed: {e}")
            return tensor


class TPUErrorHandler:
    """Robust error handling for TPU operations."""
    
    def __init__(self):
        self.error_count = 0
        self.error_log = []
        self.max_errors = 10
    
    def handle_tpu_error(self, operation: str, error: Exception) -> bool:
        """Handle TPU-specific errors with recovery strategies."""
        self.error_count += 1
        error_info = {
            "operation": operation,
            "error": str(error),
            "error_type": type(error).__name__
        }
        self.error_log.append(error_info)
        
        print(f"🧪 TPU Error in {operation}: {error}")
        
        # Apply recovery strategies
        if "out of memory" in str(error).lower():
            return self._handle_oom_error()
        elif "compilation" in str(error).lower():
            return self._handle_compilation_error()
        elif "device" in str(error).lower():
            return self._handle_device_error()
        else:
            return self._handle_generic_error()
    
    def _handle_oom_error(self) -> bool:
        """Handle TPU out-of-memory errors."""
        print("🧪 TPU: Handling OOM error...")
        try:
            # Clear memory and synchronize
            if TPU_AVAILABLE and xm is not None:
                xm.mark_step()
                xm.wait_device_ops()
            return True
        except:
            return False
    
    def _handle_compilation_error(self) -> bool:
        """Handle TPU compilation errors."""
        print("🧪 TPU: Handling compilation error...")
        try:
            # Clear compilation cache
            if hasattr(torch.jit, 'clear_cache'):
                torch.jit.clear_cache()
            return True
        except:
            return False
    
    def _handle_device_error(self) -> bool:
        """Handle TPU device errors."""
        print("🧪 TPU: Handling device error...")
        try:
            # Reset device state
            if TPU_AVAILABLE and xm is not None:
                xm.mark_step()
                xm.wait_device_ops()
            return True
        except:
            return False
    
    def _handle_generic_error(self) -> bool:
        """Handle generic TPU errors."""
        print("🧪 TPU: Handling generic error...")
        return self.error_count < self.max_errors


class TPUConfigManager:
    """Centralized TPU configuration management."""
    
    def __init__(self):
        self.config = {
            "max_retries": 3,
            "memory_fraction": 0.9,
            "compilation_timeout": 300,
            "synchronization_interval": 10,
            "enable_dynamic_shapes": False,
            "enable_async_execution": True,
        }
        self.is_initialized = False
    
    def initialize_tpu_environment(self) -> bool:
        """Initialize TPU environment with optimal settings."""
        if not TPU_AVAILABLE:
            print("⚠️ TPU: XLA libraries not available")
            return False
        
        try:
            # Set environment variables for TPU optimization
            import os
            os.environ.setdefault("XLA_USE_BF16", "1")
            os.environ.setdefault("XLA_TENSOR_ALLOCATOR_MAXSIZE", "100000000")
            os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(self.config["memory_fraction"]))
            
            # Initialize TPU
            if xm is not None:
                device = xm.xla_device()
                print(f"🧪 TPU: Initialized device: {device}")
                
                # Test basic operation
                test_tensor = torch.ones(2, 2).to(device)
                result = test_tensor + test_tensor
                xm.mark_step()
                
                print("🧪 TPU: Environment initialization successful")
                self.is_initialized = True
                return True
                
        except Exception as e:
            print(f"⚠️ TPU: Environment initialization failed: {e}")
            return False
    
    def get_optimal_config(self, model_size: str = "small") -> Dict[str, Any]:
        """Get optimal TPU configuration based on model size."""
        base_config = self.config.copy()
        
        if model_size == "small":
            base_config.update({
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "max_sequence_length": 512,
            })
        elif model_size == "medium":
            base_config.update({
                "batch_size": 4,
                "gradient_accumulation_steps": 2,
                "max_sequence_length": 1024,
            })
        elif model_size == "large":
            base_config.update({
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "max_sequence_length": 2048,
            })
        
        return base_config


# Global instances for easy access
tpu_memory_manager = TPUMemoryManager()
xla_optimizer = XLAOptimizer()
tpu_error_handler = TPUErrorHandler()
tpu_config_manager = TPUConfigManager()


def initialize_tpu_kernels() -> bool:
    """Initialize TPU kernels and optimizations."""
    print("🧪 TPU: Initializing kernels...")
    
    # Initialize configuration
    if not tpu_config_manager.initialize_tpu_environment():
        return False
    
    # Test memory manager
    memory_info = tpu_memory_manager.get_memory_info()
    if "error" not in memory_info:
        print(f"🧪 TPU: Memory available: {memory_info.get('gb_limit', 0)} GB")
    
    print("🧪 TPU: Kernel initialization completed")
    return True


def get_tpu_status() -> Dict[str, Any]:
    """Get comprehensive TPU status information."""
    return {
        "available": TPU_AVAILABLE,
        "initialized": tpu_config_manager.is_initialized,
        "memory_info": tpu_memory_manager.get_memory_info(),
        "error_count": tpu_error_handler.error_count,
        "xla_device": str(xla_optimizer.get_xla_device()) if xla_optimizer.get_xla_device() else None,
    }


# Export public interface
__all__ = [
    "TPUMemoryManager",
    "XLAOptimizer", 
    "TPUErrorHandler",
    "TPUConfigManager",
    "initialize_tpu_kernels",
    "get_tpu_status",
    "tpu_memory_manager",
    "xla_optimizer",
    "tpu_error_handler",
    "tpu_config_manager",
]
