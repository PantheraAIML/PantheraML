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

import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any
import logging
from contextlib import contextmanager

# Experimental TPU support
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

__all__ = [
    "setup_multi_gpu",
    "setup_multi_tpu",  # New experimental TPU support
    "cleanup_multi_gpu", 
    "cleanup_multi_tpu",  # New experimental TPU support
    "is_distributed_available",
    "is_tpu_available",  # New experimental TPU support
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "is_main_process",
    "MultiGPUConfig",
    "MultiTPUConfig",  # New experimental TPU config
    "distributed_context",
]

logger = logging.getLogger(__name__)


class MultiGPUConfig:
    """Configuration for multi-GPU training with PantheraML."""
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
        timeout_minutes: int = 30,
        auto_device_map: bool = True,
        tensor_parallel: bool = False,
        pipeline_parallel: bool = False,
        use_gradient_checkpointing: bool = True,
        find_unused_parameters: bool = False,
    ):
        self.backend = backend
        self.init_method = init_method
        self.timeout_minutes = timeout_minutes
        self.auto_device_map = auto_device_map
        self.tensor_parallel = tensor_parallel
        self.pipeline_parallel = pipeline_parallel
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.find_unused_parameters = find_unused_parameters


# New experimental TPU configuration class
class MultiTPUConfig:
    """Configuration for TPU training with PantheraML."""
    
    def __init__(
        self,
        num_cores: int = 8,
        auto_device_map: bool = True,
        use_gradient_checkpointing: bool = True,
    ):
        self.num_cores = num_cores
        self.auto_device_map = auto_device_map
        self.use_gradient_checkpointing = use_gradient_checkpointing


def is_distributed_available() -> bool:
    """Check if distributed training is available."""
    return dist.is_available() and torch.cuda.device_count() > 1


# New experimental TPU availability check
def is_tpu_available() -> bool:
    """Check if TPU support is available."""
    return TPU_AVAILABLE


def get_rank() -> int:
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Get the local rank within the node."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def get_tpu_rank() -> int:
    """Get the rank of the current TPU process (experimental)."""
    if TPU_AVAILABLE:
        try:
            return xm.get_ordinal()
        except:
            return 0
    return 0


def get_tpu_world_size() -> int:
    """Get the total number of TPU cores (experimental)."""
    if TPU_AVAILABLE:
        try:
            return xm.xrt_world_size()
        except:
            return 1
    return 1


def is_tpu_main_process() -> bool:
    """Check if this is the main TPU process (experimental)."""
    return get_tpu_rank() == 0


def setup_multi_gpu(config: Optional[MultiGPUConfig] = None) -> MultiGPUConfig:
    """
    Setup multi-GPU training environment.
    
    Args:
        config: MultiGPUConfig object. If None, default config is used.
        
    Returns:
        MultiGPUConfig: The configuration used for setup.
    """
    if config is None:
        config = MultiGPUConfig()
    
    # Check if we're in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        # Initialize the process group
        if not dist.is_initialized():
            if config.init_method is None:
                config.init_method = "env://"
            
            timeout = torch.distributed.default_pg_timeout
            if config.timeout_minutes > 0:
                timeout = torch.timedelta(minutes=config.timeout_minutes)
                
            dist.init_process_group(
                backend=config.backend,
                init_method=config.init_method,
                rank=rank,
                world_size=world_size,
                timeout=timeout,
            )
            
            if is_main_process():
                logger.info(f"Initialized distributed training with {world_size} GPUs")
                logger.info(f"Backend: {config.backend}")
                logger.info(f"Rank: {rank}, Local rank: {local_rank}")
    
    elif torch.cuda.device_count() > 1:
        # Local multi-GPU setup (single node)
        if not dist.is_initialized():
            dist.init_process_group(
                backend=config.backend,
                init_method="tcp://localhost:12355",
                rank=0,
                world_size=1,
            )
            logger.info(f"Setup local multi-GPU training with {torch.cuda.device_count()} GPUs")
    
    return config


# New experimental TPU setup function
def setup_multi_tpu(config: Optional[MultiTPUConfig] = None) -> MultiTPUConfig:
    """
    Setup TPU training environment.
    
    Args:
        config: MultiTPUConfig object. If None, default config is used.
        
    Returns:
        MultiTPUConfig: The configuration used for setup.
    """
    if config is None:
        config = MultiTPUConfig()
    
    if not TPU_AVAILABLE:
        raise RuntimeError("TPU support is not available. Please install torch_xla.")
    
    # Initialize the TPU system
    xmp.spawn(_init_tpu, args=(config,), nprocs=config.num_cores)
    
    return config


# New experimental TPU initialization function
def _init_tpu(rank, config: MultiTPUConfig):
    """Initialize TPU for a single process."""
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    # Initialize the process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend="xla",
            init_method="env://",
            rank=rank,
            world_size=config.num_cores,
        )
        
        logger.info(f"Initialized TPU training: Rank {rank} on TPU core {rank}")


def cleanup_multi_gpu():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        if is_main_process():
            logger.info("Cleaned up distributed training environment")


# New experimental TPU cleanup function
def cleanup_multi_tpu():
    """Cleanup TPU training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up TPU training environment")


@contextmanager
def distributed_context(config: Optional[MultiGPUConfig] = None):
    """Context manager for distributed training setup and cleanup."""
    config = setup_multi_gpu(config)
    try:
        yield config
    finally:
        cleanup_multi_gpu()


# New experimental TPU context manager
@contextmanager
def tpu_context(config: Optional[MultiTPUConfig] = None):
    """Context manager for TPU training setup and cleanup."""
    config = setup_multi_tpu(config)
    try:
        yield config
    finally:
        cleanup_multi_tpu()


def create_device_map(model_config: Dict[str, Any], num_gpus: int) -> Dict[str, int]:
    """
    Create an automatic device map for model layers across GPUs.
    
    Args:
        model_config: Model configuration dictionary
        num_gpus: Number of available GPUs
        
    Returns:
        Dict mapping layer names to GPU device IDs
    """
    device_map = {}
    
    # Get model parameters
    if hasattr(model_config, 'num_hidden_layers'):
        num_layers = model_config.num_hidden_layers
    elif 'num_hidden_layers' in model_config:
        num_layers = model_config['num_hidden_layers']
    else:
        # Fallback - try to estimate
        num_layers = 32
    
    # Distribute layers across GPUs
    layers_per_gpu = num_layers // num_gpus
    remainder = num_layers % num_gpus
    
    current_layer = 0
    for gpu_id in range(num_gpus):
        # Calculate how many layers for this GPU
        layers_this_gpu = layers_per_gpu + (1 if gpu_id < remainder else 0)
        
        # Assign layers to this GPU
        for i in range(layers_this_gpu):
            layer_name = f"model.layers.{current_layer}"
            device_map[layer_name] = gpu_id
            current_layer += 1
    
    # Place embedding and output layers
    device_map["model.embed_tokens"] = 0
    device_map["model.norm"] = num_gpus - 1
    device_map["lm_head"] = num_gpus - 1
    
    return device_map


def get_optimal_device_map(
    model, 
    num_gpus: Optional[int] = None,
    max_memory: Optional[Dict[int, str]] = None
) -> str:
    """
    Get optimal device mapping strategy for multi-GPU setup.
    
    Args:
        model: The model to be distributed
        num_gpus: Number of GPUs to use. If None, uses all available.
        max_memory: Dictionary specifying memory limits per GPU
        
    Returns:
        Device map strategy string or dictionary
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        return "auto"
    
    # For larger models, use sequential mapping
    # For smaller models that fit on one GPU, use balanced mapping
    try:
        model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024**3)  # Rough size in GB
        
        if model_size > 10:  # Large models
            return "sequential"
        else:  # Smaller models
            return "balanced"
    except:
        return "sequential"


def wrap_model_for_distributed(
    model,
    config: MultiGPUConfig,
    device_ids: Optional[list] = None,
) -> torch.nn.Module:
    """
    Wrap model for distributed training.
    
    Args:
        model: The model to wrap
        config: Multi-GPU configuration
        device_ids: List of device IDs to use
        
    Returns:
        Wrapped model ready for distributed training
    """
    if not dist.is_initialized():
        return model
    
    if device_ids is None:
        device_ids = [get_local_rank()]
    
    # Use DistributedDataParallel for multi-GPU training
    if get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=device_ids[0],
            find_unused_parameters=config.find_unused_parameters,
        )
        
        if is_main_process():
            logger.info(f"Wrapped model with DistributedDataParallel")
    
    elif torch.cuda.device_count() > 1:
        # Single node multi-GPU
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        
        if is_main_process():
            logger.info(f"Wrapped model with DataParallel for {len(device_ids)} GPUs")
    
    return model


def get_sampler_for_distributed(dataset, shuffle: bool = True):
    """Get appropriate sampler for distributed training."""
    if get_world_size() > 1:
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
        )
    return None


def all_reduce_scalar(scalar: float, average: bool = True) -> float:
    """All-reduce a scalar value across all processes."""
    if not dist.is_initialized() or get_world_size() <= 1:
        return scalar
    
    tensor = torch.tensor(scalar, device=torch.cuda.current_device())
    dist.all_reduce(tensor)
    
    if average:
        tensor /= get_world_size()
    
    return tensor.item()


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(obj, src: int = 0):
    """Broadcast an object from src rank to all other ranks."""
    if not dist.is_initialized():
        return obj
    
    if get_rank() == src:
        objects = [obj]
    else:
        objects = [None]
    
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def get_tpu_device() -> torch.device:
    """
    Get the current TPU device (EXPERIMENTAL).
    
    Returns:
        torch.device: The TPU device.
    """
    if not TPU_AVAILABLE:
        raise RuntimeError("TPU support requires torch_xla")
    
    try:
        return xm.xla_device()
    except Exception as e:
        raise RuntimeError(f"Failed to get TPU device: {e}")


def synchronize_tpu():
    """
    Synchronize TPU operations (EXPERIMENTAL).
    """
    if TPU_AVAILABLE:
        try:
            xm.mark_step()
        except Exception as e:
            logging.warning(f"TPU synchronization warning: {e}")
