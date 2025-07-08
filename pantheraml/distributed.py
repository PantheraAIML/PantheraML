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

import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any, Tuple
import logging
import warnings
from contextlib import contextmanager

# Enhanced TPU support with Phase 1 improvements
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr
    from .kernels.tpu_kernels import (
        tpu_memory_manager, 
        xla_optimizer, 
        tpu_error_handler, 
        tpu_config_manager,
        initialize_tpu_kernels
    )
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    xm = None
    pl = None
    xmp = None
    xr = None

# Phase 2 TPU Performance Integration
try:
    from .kernels.tpu_performance import (
        TPUCommunicationOptimizer, ModelShardManager, TPUPerformanceProfiler
    )
    PHASE2_TPU_AVAILABLE = True
except ImportError:
    PHASE2_TPU_AVAILABLE = False

# Phase 3 Advanced Distributed Training
try:
    from .kernels.tpu_advanced import (
        Phase3Manager, MultiPodConfig, JAXConfig, AutoScalingConfig
    )
    PHASE3_TPU_AVAILABLE = True
except ImportError:
    PHASE3_TPU_AVAILABLE = False

__all__ = [
    "setup_multi_gpu",
    "setup_multi_tpu",  # Enhanced TPU support with Phase 1
    "cleanup_multi_gpu", 
    "cleanup_multi_tpu",  # Enhanced TPU cleanup with Phase 1
    "is_distributed_available",
    "is_tpu_available",  # TPU availability check
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "is_main_process",
    "get_tpu_rank",  # TPU rank functions
    "get_tpu_world_size",
    "is_tpu_main_process",
    "get_tpu_device",  # Phase 1: Enhanced TPU device management
    "synchronize_tpu",  # Phase 1: Enhanced TPU synchronization
    "get_tpu_memory_info",  # Phase 1: TPU memory tracking
    "optimize_tpu_memory",  # Phase 1: TPU memory optimization
    "get_tpu_status",  # Phase 1: Comprehensive TPU status
    "MultiGPUConfig",
    "MultiTPUConfig",  # Enhanced TPU config
    "distributed_context",
    "tpu_context",  # Phase 1: TPU context manager
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


# Enhanced TPU setup function with Phase 1 improvements
def setup_multi_tpu(config: Optional[MultiTPUConfig] = None) -> MultiTPUConfig:
    """
    Setup TPU training environment with enhanced stability and error handling.
    
    Args:
        config: MultiTPUConfig object. If None, default config is used.
        
    Returns:
        MultiTPUConfig: The configuration used for setup.
    """
    if config is None:
        config = MultiTPUConfig()
    
    if not TPU_AVAILABLE:
        print("⚠️ TPU: XLA libraries not available. Please install torch_xla.")
        raise RuntimeError(
            "TPU support requires torch_xla. Install with: pip install torch_xla"
        )
    
    try:
        # Phase 1: Initialize TPU kernels and optimizations
        print("🧪 TPU: Initializing Phase 1 enhancements...")
        
        # Initialize TPU configuration manager
        if not tpu_config_manager.initialize_tpu_environment():
            raise RuntimeError("Failed to initialize TPU environment")
        
        # Initialize TPU kernels
        if not initialize_tpu_kernels():
            print("⚠️ TPU: Kernel initialization failed, continuing with basic setup")
        
        # Get optimal configuration for TPU
        optimal_config = tpu_config_manager.get_optimal_config("medium")
        print(f"🧪 TPU: Using optimal config: {optimal_config}")
        
        # Set up error handling
        tpu_error_handler.error_count = 0
        
        # Initialize XLA optimizer
        device = xla_optimizer.get_xla_device()
        if device is None:
            raise RuntimeError("Failed to get XLA device")
        
        print(f"🧪 TPU: XLA device initialized: {device}")
        
        # Memory optimization
        memory_info = tpu_memory_manager.get_memory_info()
        if "error" not in memory_info:
            print(f"🧪 TPU: Memory available: {memory_info.get('gb_limit', 0)} GB")
            tpu_memory_manager.optimize_memory()
        
        # Initialize the TPU system with error handling
        try:
            xmp.spawn(_init_tpu_enhanced, args=(config,), nprocs=config.num_cores)
        except Exception as e:
            if tpu_error_handler.handle_tpu_error("spawn_initialization", e):
                # Retry with fallback configuration
                print("🧪 TPU: Retrying with fallback configuration...")
                config.num_cores = min(config.num_cores, 4)  # Reduce cores as fallback
                xmp.spawn(_init_tpu_enhanced, args=(config,), nprocs=config.num_cores)
            else:
                raise
        
        print("✅ TPU: Enhanced setup completed successfully")
        return config
        
    except Exception as e:
        error_msg = f"TPU setup failed: {e}"
        print(f"❌ {error_msg}")
        if tpu_error_handler.handle_tpu_error("setup_multi_tpu", e):
            print("🔄 TPU: Attempting fallback to single-core mode...")
            config.num_cores = 1
            return config
        else:
            raise RuntimeError(error_msg)


# Enhanced TPU initialization function with Phase 1 improvements
def _init_tpu_enhanced(rank, config: MultiTPUConfig):
    """Initialize TPU for a single process with enhanced error handling."""
    try:
        print(f"🧪 TPU Core {rank}: Starting enhanced initialization...")
        
        # Get XLA device for this core
        device = xm.xla_device()
        print(f"🧪 TPU Core {rank}: XLA device: {device}")
        
        # Apply memory optimizations
        tpu_memory_manager.optimize_memory()
        
        # Test basic operations to ensure TPU is working
        test_tensor = torch.ones(2, 2, device=device)
        result = test_tensor + test_tensor
        xm.mark_step()  # Synchronize TPU operations
        
        # Initialize process group with error handling
        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend="xla",
                    init_method="env://",
                    rank=rank,
                    world_size=config.num_cores,
                    timeout=torch.timedelta(minutes=10),  # Increased timeout
                )
                print(f"✅ TPU Core {rank}: Process group initialized")
            except Exception as e:
                print(f"⚠️ TPU Core {rank}: Process group init failed: {e}")
                # Continue without distributed setup for single-core fallback
        
        # Memory status check
        memory_info = tpu_memory_manager.get_memory_info()
        if "error" not in memory_info:
            print(f"🧪 TPU Core {rank}: Memory usage: {memory_info.get('gb_in_use', 0):.2f}/{memory_info.get('gb_limit', 0):.2f} GB")
        
        print(f"✅ TPU Core {rank}: Enhanced initialization completed")
        
    except Exception as e:
        error_msg = f"TPU Core {rank} initialization failed: {e}"
        print(f"❌ {error_msg}")
        if not tpu_error_handler.handle_tpu_error(f"init_tpu_core_{rank}", e):
            raise RuntimeError(error_msg)


def cleanup_multi_gpu():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        if is_main_process():
            logger.info("Cleaned up distributed training environment")


# Enhanced TPU cleanup with Phase 1 improvements
def cleanup_multi_tpu():
    """Enhanced TPU training environment cleanup with proper error handling."""
    try:
        print("🧪 TPU: Starting enhanced cleanup...")
        
        # Clear TPU memory cache
        if TPU_AVAILABLE and tpu_memory_manager:
            tpu_memory_manager.clear_cache()
            print("🧪 TPU: Memory cache cleared")
        
        # Synchronize all TPU operations
        if TPU_AVAILABLE and xm:
            xm.mark_step()
            xm.wait_device_ops()
            print("🧪 TPU: Operations synchronized")
        
        # Cleanup distributed process group
        if dist.is_initialized():
            dist.destroy_process_group()
            print("🧪 TPU: Process group destroyed")
        
        # Reset error handler
        if TPU_AVAILABLE and tpu_error_handler:
            tpu_error_handler.error_count = 0
            tpu_error_handler.error_log.clear()
        
        print("✅ TPU: Enhanced cleanup completed")
        
    except Exception as e:
        print(f"⚠️ TPU: Cleanup warning: {e}")
        # Don't raise exception during cleanup


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


def get_tpu_device() -> Optional[torch.device]:
    """Get the current TPU device with error handling."""
    if not TPU_AVAILABLE or not xm:
        return None
    
    try:
        return xm.xla_device()
    except Exception as e:
        if tpu_error_handler:
            tpu_error_handler.handle_tpu_error("get_tpu_device", e)
        return None


def synchronize_tpu():
    """Synchronize TPU operations with error handling."""
    if not TPU_AVAILABLE or not xm:
        return False
    
    try:
        xm.mark_step()
        xm.wait_device_ops()
        return True
    except Exception as e:
        if tpu_error_handler:
            tpu_error_handler.handle_tpu_error("synchronize_tpu", e)
        return False


def get_tpu_memory_info() -> Dict[str, Any]:
    """Get TPU memory information with enhanced error handling."""
    if not TPU_AVAILABLE or not tpu_memory_manager:
        return {"error": "TPU not available"}
    
    return tpu_memory_manager.get_memory_info()


def optimize_tpu_memory() -> bool:
    """Optimize TPU memory usage."""
    if not TPU_AVAILABLE or not tpu_memory_manager:
        return False
    
    return tpu_memory_manager.optimize_memory()


def get_tpu_status() -> Dict[str, Any]:
    """Get comprehensive TPU status information."""
    if not TPU_AVAILABLE:
        return {
            "available": False,
            "error": "TPU libraries not installed"
        }
    
    try:
        status = {
            "available": True,
            "initialized": tpu_config_manager.is_initialized if tpu_config_manager else False,
            "device": str(get_tpu_device()) if get_tpu_device() else None,
            "memory_info": get_tpu_memory_info(),
            "error_count": tpu_error_handler.error_count if tpu_error_handler else 0,
            "rank": get_tpu_rank(),
            "world_size": get_tpu_world_size(),
            "is_main_process": is_tpu_main_process(),
        }
        
        # Add XLA compilation info
        if xla_optimizer:
            status["compiled_functions"] = len(xla_optimizer.compiled_functions)
        
        return status
        
    except Exception as e:
        return {
            "available": True,
            "error": f"Status check failed: {e}"
        }


def setup_phase2_distributed_training(
    world_size: int = None,
    enable_sharding: bool = True,
    enable_comm_optimization: bool = True,
    enable_profiling: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Setup Phase 2 distributed training with advanced TPU optimizations.
    
    Args:
        world_size: Number of TPU cores/devices
        enable_sharding: Enable model sharding across devices
        enable_comm_optimization: Enable communication optimizations
        enable_profiling: Enable detailed performance profiling
        **kwargs: Additional configuration options
    
    Returns:
        Dictionary with Phase 2 components and configuration
    """
    if not TPU_AVAILABLE:
        warnings.warn("TPU not available, Phase 2 setup skipped")
        return {}
    
    if not PHASE2_TPU_AVAILABLE:
        warnings.warn("Phase 2 components not available, using Phase 1 only")
        return {}
    
    phase2_config = {}
    
    try:
        # Initialize communication optimizer
        if enable_comm_optimization:
            comm_optimizer = TPUCommunicationOptimizer()
            comm_optimizer.setup_distributed_communication(world_size or xm.xrt_world_size())
            phase2_config['comm_optimizer'] = comm_optimizer
            print(f"✅ Phase 2 communication optimization enabled for {world_size or xm.xrt_world_size()} devices")
        
        # Initialize model sharding manager
        if enable_sharding:
            shard_manager = ModelShardManager(
                num_shards=world_size or xm.xrt_world_size(),
                shard_axis=kwargs.get('shard_axis', 0)
            )
            phase2_config['shard_manager'] = shard_manager
            print(f"✅ Phase 2 model sharding enabled with {shard_manager.num_shards} shards")
        
        # Initialize performance profiler
        if enable_profiling:
            profiler = TPUPerformanceProfiler(enable_detailed=True)
            profiler.start_distributed_profiling()
            phase2_config['profiler'] = profiler
            print("✅ Phase 2 distributed profiling enabled")
        
        return phase2_config
        
    except Exception as e:
        warnings.warn(f"Phase 2 distributed setup failed: {e}")
        return {}


def optimize_distributed_communication(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    phase2_config: Dict[str, Any] = None
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Apply Phase 2 communication optimizations to model and optimizer.
    
    Args:
        model: The model to optimize
        optimizer: The optimizer to optimize
        phase2_config: Phase 2 configuration from setup
    
    Returns:
        Optimized model and optimizer
    """
    if not TPU_AVAILABLE or not PHASE2_TPU_AVAILABLE:
        return model, optimizer
    
    if not phase2_config:
        return model, optimizer
    
    try:
        # Apply communication optimizations
        comm_optimizer = phase2_config.get('comm_optimizer')
        if comm_optimizer:
            model = comm_optimizer.optimize_model_communication(model)
            optimizer = comm_optimizer.optimize_optimizer_communication(optimizer)
            print("✅ Communication optimizations applied to model and optimizer")
        
        # Apply model sharding
        shard_manager = phase2_config.get('shard_manager')
        if shard_manager:
            model = shard_manager.shard_model(model)
            print(f"✅ Model sharded across {shard_manager.num_shards} devices")
        
        return model, optimizer
        
    except Exception as e:
        warnings.warn(f"Communication optimization failed: {e}")
        return model, optimizer


def synchronize_phase2_training(
    loss: torch.Tensor,
    phase2_config: Dict[str, Any] = None,
    step: int = 0
) -> torch.Tensor:
    """
    Synchronize training step with Phase 2 optimizations.
    
    Args:
        loss: Training loss tensor
        phase2_config: Phase 2 configuration
        step: Current training step
    
    Returns:
        Synchronized loss tensor
    """
    if not TPU_AVAILABLE:
        return loss
    
    try:
        # Standard XLA synchronization
        xm.mark_step()
        
        # Phase 2 communication optimization
        if phase2_config and 'comm_optimizer' in phase2_config:
            comm_optimizer = phase2_config['comm_optimizer']
            loss = comm_optimizer.synchronize_gradients(loss)
        
        # Phase 2 profiling
        if phase2_config and 'profiler' in phase2_config:
            profiler = phase2_config['profiler']
            profiler.record_step_metrics(step, loss.item())
        
        return loss
        
    except Exception as e:
        warnings.warn(f"Phase 2 synchronization warning: {e}")
        return loss


def cleanup_phase2_distributed(phase2_config: Dict[str, Any] = None):
    """
    Cleanup Phase 2 distributed training components.
    
    Args:
        phase2_config: Phase 2 configuration to cleanup
    """
    if not phase2_config:
        return
    
    try:
        # Cleanup communication optimizer
        if 'comm_optimizer' in phase2_config:
            phase2_config['comm_optimizer'].cleanup()
            print("✅ Communication optimizer cleaned up")
        
        # Cleanup shard manager
        if 'shard_manager' in phase2_config:
            phase2_config['shard_manager'].cleanup()
            print("✅ Shard manager cleaned up")
        
        # Finalize profiler
        if 'profiler' in phase2_config:
            phase2_config['profiler'].finalize()
            print("✅ Profiler finalized")
        
        print("✅ Phase 2 distributed cleanup completed")
        
    except Exception as e:
        warnings.warn(f"Phase 2 cleanup warning: {e}")


# Enhanced wrapper function with Phase 2 support
def setup_enhanced_distributed_training(
    model: torch.nn.Module,
    enable_phase2: bool = True,
    **kwargs
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Setup enhanced distributed training with Phase 1 and Phase 2 support.
    
    Args:
        model: Model to setup for distributed training
        enable_phase2: Whether to enable Phase 2 optimizations
        **kwargs: Additional configuration options
    
    Returns:
        Tuple of (prepared_model, config_dict)
    """
    # Phase 1 setup (always enabled if TPU available)
    phase1_config = setup_multi_tpu() if TPU_AVAILABLE else {}
    
    # Phase 2 setup (if enabled and available)
    phase2_config = {}
    if enable_phase2:
        phase2_config = setup_phase2_distributed_training(**kwargs)
    
    # Combine configurations
    combined_config = {
        'phase1': phase1_config,
        'phase2': phase2_config,
        'phase2_enabled': enable_phase2 and bool(phase2_config)
    }
    
    # Apply Phase 2 optimizations if available
    if combined_config['phase2_enabled']:
        model, _ = optimize_distributed_communication(model, None, phase2_config)
    
    return model, combined_config


def setup_phase3_advanced_training(
    model: torch.nn.Module,
    enable_multi_pod: bool = False,
    enable_jax_backend: bool = True,
    enable_auto_scaling: bool = False,
    num_pods: int = 1,
    cores_per_pod: int = 8,
    **kwargs
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Setup Phase 3 advanced TPU training with cutting-edge features.
    
    Args:
        model: Model to setup for advanced training
        enable_multi_pod: Enable multi-pod TPU coordination
        enable_jax_backend: Enable JAX/Flax native backend
        enable_auto_scaling: Enable dynamic resource scaling
        num_pods: Number of TPU pods
        cores_per_pod: Cores per TPU pod
        **kwargs: Additional configuration options
    
    Returns:
        Tuple of (prepared_model, phase3_config)
    """
    if not TPU_AVAILABLE:
        warnings.warn("TPU not available, Phase 3 setup skipped")
        return model, {}
    
    if not PHASE3_TPU_AVAILABLE:
        warnings.warn("Phase 3 components not available")
        return model, {}
    
    phase3_config = {}
    
    try:
        print("🚀 Setting up Phase 3 advanced TPU training...")
        
        # Multi-pod configuration
        multi_pod_config = MultiPodConfig(
            num_pods=num_pods,
            cores_per_pod=cores_per_pod,
            enable_pod_communication=enable_multi_pod,
            enable_fault_tolerance=kwargs.get('enable_fault_tolerance', True)
        )
        
        # JAX/Flax configuration
        jax_config = JAXConfig(
            enable_jax_backend=enable_jax_backend,
            precision=kwargs.get('precision', 'bfloat16'),
            mesh_shape=kwargs.get('mesh_shape', (1, cores_per_pod)),
            use_pmap=kwargs.get('use_pmap', True),
            use_jit=kwargs.get('use_jit', True)
        )
        
        # Auto-scaling configuration
        auto_scaling_config = AutoScalingConfig(
            enable_auto_scaling=enable_auto_scaling,
            min_cores=kwargs.get('min_cores', 8),
            max_cores=kwargs.get('max_cores', num_pods * cores_per_pod * 4),
            scale_up_threshold=kwargs.get('scale_up_threshold', 0.85),
            scale_down_threshold=kwargs.get('scale_down_threshold', 0.4)
        )
        
        # Initialize Phase 3 manager
        phase3_manager = Phase3Manager(
            multi_pod_config=multi_pod_config,
            jax_config=jax_config,
            auto_scaling_config=auto_scaling_config
        )
        
        phase3_config = {
            'phase3_manager': phase3_manager,
            'multi_pod_config': multi_pod_config,
            'jax_config': jax_config,
            'auto_scaling_config': auto_scaling_config,
            'capabilities': {
                'multi_pod': enable_multi_pod and num_pods > 1,
                'jax_backend': enable_jax_backend,
                'auto_scaling': enable_auto_scaling,
                'fault_tolerance': True
            }
        }
        
        print("✅ Phase 3 advanced training setup completed")
        print(f"🌟 Capabilities: {list(phase3_config['capabilities'].keys())}")
        
        return model, phase3_config
        
    except Exception as e:
        warnings.warn(f"Phase 3 setup failed: {e}")
        return model, {}


def coordinate_multi_pod_training(
    training_fn: Callable,
    phase3_config: Dict[str, Any],
    *args, **kwargs
) -> Any:
    """
    Coordinate training across multiple TPU pods.
    
    Args:
        training_fn: Training function to coordinate
        phase3_config: Phase 3 configuration
        *args, **kwargs: Arguments for training function
    
    Returns:
        Training result with multi-pod coordination
    """
    if not phase3_config or 'phase3_manager' not in phase3_config:
        return training_fn(*args, **kwargs)
    
    try:
        phase3_manager = phase3_config['phase3_manager']
        
        if phase3_manager.multi_pod_coordinator:
            print("🌐 Coordinating training across multiple TPU pods...")
            
            # Apply multi-pod optimizations
            optimized_fn = phase3_manager.optimize_training_step(training_fn)
            
            # Execute with coordination
            result = optimized_fn(*args, **kwargs)
            
            print("✅ Multi-pod training coordination completed")
            return result
        else:
            return training_fn(*args, **kwargs)
            
    except Exception as e:
        print(f"⚠️ Multi-pod coordination failed: {e}")
        return training_fn(*args, **kwargs)


def enable_jax_acceleration(
    model: torch.nn.Module,
    phase3_config: Dict[str, Any]
) -> torch.nn.Module:
    """
    Enable JAX/Flax acceleration for native TPU performance.
    
    Args:
        model: PyTorch model to accelerate
        phase3_config: Phase 3 configuration
    
    Returns:
        JAX-accelerated model (or original if conversion fails)
    """
    if not phase3_config or 'phase3_manager' not in phase3_config:
        return model
    
    try:
        phase3_manager = phase3_config['phase3_manager']
        
        if phase3_manager.jax_integration:
            print("⚡ Enabling JAX/Flax acceleration for native TPU performance...")
            
            # Convert to JAX/Flax
            jax_model = phase3_manager.jax_integration.torch_to_jax_model(model)
            
            if jax_model:
                print("✅ JAX/Flax acceleration enabled")
                # Mark original model as JAX-ready
                model._jax_model = jax_model
                model._jax_enabled = True
                return model
            else:
                print("⚠️ JAX conversion failed, using PyTorch model")
                return model
        else:
            return model
            
    except Exception as e:
        print(f"⚠️ JAX acceleration failed: {e}")
        return model


def monitor_auto_scaling(
    phase3_config: Dict[str, Any],
    training_metrics: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Monitor and manage auto-scaling during training.
    
    Args:
        phase3_config: Phase 3 configuration
        training_metrics: Current training metrics
    
    Returns:
        Auto-scaling status and metrics
    """
    if not phase3_config or 'phase3_manager' not in phase3_config:
        return {}
    
    try:
        phase3_manager = phase3_config['phase3_manager']
        
        if phase3_manager.auto_scaling_manager:
            # Get current scaling status
            scaling_metrics = {
                'current_cores': phase3_manager.auto_scaling_manager.current_cores,
                'scaling_history': len(phase3_manager.auto_scaling_manager.scaling_history),
                'enabled': phase3_config['auto_scaling_config'].enable_auto_scaling
            }
            
            # Add training metrics if provided
            if training_metrics:
                scaling_metrics['training_metrics'] = training_metrics
            
            return scaling_metrics
        else:
            return {'auto_scaling': 'disabled'}
            
    except Exception as e:
        print(f"⚠️ Auto-scaling monitoring failed: {e}")
        return {'error': str(e)}


def get_comprehensive_phase3_metrics(
    phase3_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get comprehensive metrics from all Phase 3 components.
    
    Args:
        phase3_config: Phase 3 configuration
    
    Returns:
        Comprehensive Phase 3 metrics
    """
    if not phase3_config or 'phase3_manager' not in phase3_config:
        return {}
    
    try:
        phase3_manager = phase3_config['phase3_manager']
        metrics = phase3_manager.get_phase3_metrics()
        
        # Add configuration info
        metrics['configuration'] = {
            'multi_pod_enabled': 'multi_pod_config' in phase3_config,
            'jax_enabled': 'jax_config' in phase3_config,
            'auto_scaling_enabled': 'auto_scaling_config' in phase3_config,
            'capabilities': phase3_config.get('capabilities', {})
        }
        
        return metrics
        
    except Exception as e:
        print(f"⚠️ Phase 3 metrics collection failed: {e}")
        return {'error': str(e)}


def cleanup_phase3_training(phase3_config: Dict[str, Any]):
    """
    Cleanup Phase 3 advanced training components.
    
    Args:
        phase3_config: Phase 3 configuration to cleanup
    """
    if not phase3_config:
        return
    
    try:
        # Cleanup Phase 3 manager
        if 'phase3_manager' in phase3_config:
            phase3_manager = phase3_config['phase3_manager']
            phase3_manager.cleanup()
            print("✅ Phase 3 manager cleaned up")
        
        # Clear configuration
        phase3_config.clear()
        
        print("✅ Phase 3 advanced training cleanup completed")
        
    except Exception as e:
        warnings.warn(f"Phase 3 cleanup warning: {e}")


# Enhanced setup function that combines all phases
def setup_ultimate_distributed_training(
    model: torch.nn.Module,
    enable_all_phases: bool = True,
    **kwargs
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Setup ultimate distributed training with all phases (1, 2, 3).
    
    Args:
        model: Model to setup for ultimate training
        enable_all_phases: Enable all available phases
        **kwargs: Configuration options for all phases
    
    Returns:
        Tuple of (prepared_model, ultimate_config)
    """
    ultimate_config = {
        'phases_enabled': [],
        'configurations': {},
        'capabilities': {}
    }
    
    try:
        # Phase 1 + 2 setup (if available)
        if enable_all_phases:
            model, phase12_config = setup_enhanced_distributed_training(
                model, enable_phase2=True, **kwargs
            )
            
            if phase12_config:
                ultimate_config['phases_enabled'].extend(['phase1', 'phase2'])
                ultimate_config['configurations']['phase12'] = phase12_config
                print("✅ Phase 1 + 2 setup completed")
        
        # Phase 3 setup (if available and requested)
        if enable_all_phases and PHASE3_TPU_AVAILABLE:
            model, phase3_config = setup_phase3_advanced_training(
                model, **kwargs
            )
            
            if phase3_config:
                ultimate_config['phases_enabled'].append('phase3')
                ultimate_config['configurations']['phase3'] = phase3_config
                print("✅ Phase 3 setup completed")
        
        # Summary
        total_phases = len(ultimate_config['phases_enabled'])
        print(f"🎉 Ultimate distributed training setup: {total_phases} phases enabled")
        print(f"🚀 Enabled phases: {ultimate_config['phases_enabled']}")
        
        return model, ultimate_config
        
    except Exception as e:
        warnings.warn(f"Ultimate setup failed: {e}")
        return model, ultimate_config
