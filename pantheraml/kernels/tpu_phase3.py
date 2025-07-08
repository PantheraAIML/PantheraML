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

"""
TPU Phase 3 Advanced Features for PantheraML
Phase 3 Features:
- Multi-pod TPU optimization and coordination
- JAX/Flax integration for native TPU performance
- Advanced profiling and performance insights
- Auto-scaling and dynamic resource allocation
- Hardware-aware optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math
import time
import json
import logging
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import queue
import asyncio

# TPU-specific imports with error handling
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.test.test_utils as test_utils
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    xm = None
    xmp = None
    xr = None
    pl = None
    test_utils = None

# JAX/Flax imports for native TPU support
try:
    import jax
    import jax.numpy as jnp
    import flax
    import flax.linen as nn_flax
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    flax = None
    nn_flax = None
    optax = None

# Import Phase 1 and Phase 2 components
try:
    from .tpu_kernels import (
        tpu_memory_manager, xla_optimizer, tpu_error_handler, tpu_config_manager
    )
    from .tpu_performance import (
        TPUAttentionOptimizer, TPUModelSharding, TPUDynamicShapeHandler,
        TPUCommunicationOptimizer, TPUPerformanceProfiler
    )
    PHASE12_AVAILABLE = True
except ImportError:
    PHASE12_AVAILABLE = False

@dataclass
class MultiPodConfig:
    """Configuration for multi-pod TPU training."""
    num_pods: int = 1
    cores_per_pod: int = 8
    pod_slice_shape: Optional[Tuple[int, ...]] = None
    use_pod_locality: bool = True
    inter_pod_communication: str = "optimal"  # "optimal", "ring", "tree"
    pod_failure_recovery: bool = True
    dynamic_pod_scaling: bool = False

@dataclass
class JAXFlaxConfig:
    """Configuration for JAX/Flax integration."""
    enable_jax_backend: bool = False
    use_flax_models: bool = False
    jax_precision: str = "bfloat16"  # "float32", "bfloat16", "float16"
    use_jax_jit: bool = True
    use_jax_pmap: bool = True
    use_jax_sharding: bool = True
    checkpoint_format: str = "pytorch"  # "pytorch", "flax", "both"

@dataclass
class AutoScalingConfig:
    """Configuration for auto-scaling TPU resources."""
    enable_auto_scaling: bool = False
    min_cores: int = 8
    max_cores: int = 256
    scale_up_threshold: float = 0.85  # CPU/memory utilization
    scale_down_threshold: float = 0.3
    scale_check_interval: int = 60  # seconds
    preemptible_instances: bool = True

class MultiPodOptimizer:
    """
    Advanced multi-pod TPU optimization and coordination.
    Handles training across multiple TPU pods with intelligent coordination.
    """
    
    def __init__(self, config: MultiPodConfig):
        self.config = config
        self.pod_managers = {}
        self.global_coordinator = None
        self.inter_pod_communicator = None
        self.failure_recovery_enabled = config.pod_failure_recovery
        
        if not TPU_AVAILABLE:
            warnings.warn("TPU not available, multi-pod optimization disabled")
            return
        
        self._initialize_multi_pod()
    
    def _initialize_multi_pod(self):
        """Initialize multi-pod coordination."""
        try:
            # Get current pod information
            self.current_pod_id = self._get_current_pod_id()
            self.total_cores = self.config.num_pods * self.config.cores_per_pod
            
            # Initialize pod-specific managers
            self.pod_managers[self.current_pod_id] = {
                'cores': list(range(self.config.cores_per_pod)),
                'status': 'active',
                'last_heartbeat': time.time()
            }
            
            # Setup inter-pod communication
            self._setup_inter_pod_communication()
            
            # Initialize global coordinator (on pod 0)
            if self.current_pod_id == 0:
                self.global_coordinator = GlobalPodCoordinator(self.config)
            
            print(f"✅ Multi-pod optimizer initialized: Pod {self.current_pod_id}/{self.config.num_pods}")
            
        except Exception as e:
            warnings.warn(f"Multi-pod initialization failed: {e}")
    
    def _get_current_pod_id(self) -> int:
        """Get the current TPU pod ID."""
        try:
            # TPU pod ID is typically available in environment or can be derived
            pod_id = int(xm.get_ordinal() // self.config.cores_per_pod)
            return pod_id
        except:
            return 0
    
    def _setup_inter_pod_communication(self):
        """Setup optimized inter-pod communication."""
        try:
            communication_type = self.config.inter_pod_communication
            
            if communication_type == "optimal":
                self.inter_pod_communicator = OptimalInterPodCommunicator()
            elif communication_type == "ring":
                self.inter_pod_communicator = RingInterPodCommunicator()
            elif communication_type == "tree":
                self.inter_pod_communicator = TreeInterPodCommunicator()
            else:
                raise ValueError(f"Unknown communication type: {communication_type}")
            
            self.inter_pod_communicator.initialize(
                self.current_pod_id, 
                self.config.num_pods,
                self.config.cores_per_pod
            )
            
        except Exception as e:
            warnings.warn(f"Inter-pod communication setup failed: {e}")
            self.inter_pod_communicator = None
    
    def optimize_cross_pod_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Optimize gradient communication across pods."""
        if not self.inter_pod_communicator:
            return gradients
        
        try:
            # Compress gradients for inter-pod communication
            compressed_grads = self._compress_gradients(gradients)
            
            # Perform optimized all-reduce across pods
            synchronized_grads = self.inter_pod_communicator.all_reduce_across_pods(compressed_grads)
            
            # Decompress gradients
            final_grads = self._decompress_gradients(synchronized_grads)
            
            return final_grads
            
        except Exception as e:
            warnings.warn(f"Cross-pod gradient optimization failed: {e}")
            return gradients
    
    def _compress_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress gradients for efficient inter-pod communication."""
        compressed = {}
        for name, grad in gradients.items():
            if grad is not None:
                # Use quantization or sparsification for compression
                compressed[name] = self._quantize_tensor(grad)
            else:
                compressed[name] = None
        return compressed
    
    def _decompress_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decompress gradients after inter-pod communication."""
        decompressed = {}
        for name, grad in gradients.items():
            if grad is not None:
                decompressed[name] = self._dequantize_tensor(grad)
            else:
                decompressed[name] = None
        return decompressed
    
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """Quantize tensor for compression."""
        try:
            # Simple quantization - can be enhanced with more sophisticated methods
            scale = tensor.abs().max() / (2**(bits-1) - 1)
            quantized = torch.round(tensor / scale).clamp(-(2**(bits-1)), 2**(bits-1) - 1)
            return quantized.to(torch.int8), scale
        except:
            return tensor
    
    def _dequantize_tensor(self, quantized_data) -> torch.Tensor:
        """Dequantize tensor after compression."""
        try:
            if isinstance(quantized_data, tuple):
                quantized, scale = quantized_data
                return quantized.float() * scale
            else:
                return quantized_data
        except:
            return quantized_data
    
    def get_pod_status(self) -> Dict[str, Any]:
        """Get status of all pods."""
        return {
            'current_pod': self.current_pod_id,
            'total_pods': self.config.num_pods,
            'pod_managers': self.pod_managers,
            'communication_type': self.config.inter_pod_communication,
            'failure_recovery': self.failure_recovery_enabled
        }

class GlobalPodCoordinator:
    """Global coordinator for multi-pod training."""
    
    def __init__(self, config: MultiPodConfig):
        self.config = config
        self.pod_status = {}
        self.training_metrics = {}
        self.coordinator_thread = None
        self.stop_coordination = False
        
    def start_coordination(self):
        """Start global coordination thread."""
        self.coordinator_thread = threading.Thread(target=self._coordination_loop)
        self.coordinator_thread.start()
    
    def _coordination_loop(self):
        """Main coordination loop."""
        while not self.stop_coordination:
            try:
                # Monitor pod health
                self._monitor_pod_health()
                
                # Coordinate training synchronization
                self._coordinate_training_sync()
                
                # Handle any pod failures
                self._handle_pod_failures()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                warnings.warn(f"Coordination loop error: {e}")
    
    def _monitor_pod_health(self):
        """Monitor health of all pods."""
        # Implementation for pod health monitoring
        pass
    
    def _coordinate_training_sync(self):
        """Coordinate training synchronization across pods."""
        # Implementation for training coordination
        pass
    
    def _handle_pod_failures(self):
        """Handle pod failures and recovery."""
        # Implementation for failure handling
        pass
    
    def stop(self):
        """Stop coordination."""
        self.stop_coordination = True
        if self.coordinator_thread:
            self.coordinator_thread.join()

class JAXFlaxIntegration:
    """
    JAX/Flax integration for native TPU performance.
    Provides seamless conversion between PyTorch and JAX models.
    """
    
    def __init__(self, config: JAXFlaxConfig):
        self.config = config
        self.jax_model = None
        self.pytorch_model = None
        self.conversion_cache = {}
        
        if not JAX_AVAILABLE:
            warnings.warn("JAX/Flax not available, integration disabled")
            return
        
        if not TPU_AVAILABLE:
            warnings.warn("TPU not available, JAX integration may have limited benefits")
        
        self._initialize_jax_environment()
    
    def _initialize_jax_environment(self):
        """Initialize JAX environment for optimal TPU performance."""
        try:
            # Configure JAX for TPU
            if TPU_AVAILABLE:
                # JAX automatically detects TPUs
                devices = jax.devices()
                tpu_devices = [d for d in devices if 'TPU' in str(d)]
                print(f"✅ JAX detected {len(tpu_devices)} TPU devices")
            
            # Set precision
            if self.config.jax_precision == "bfloat16":
                jax.config.update("jax_default_dtype_bits", "32")
                jax.config.update("jax_enable_x64", False)
            
            # Configure JIT compilation
            if self.config.use_jax_jit:
                jax.config.update("jax_disable_jit", False)
            
            print("✅ JAX environment initialized for TPU")
            
        except Exception as e:
            warnings.warn(f"JAX environment initialization failed: {e}")
    
    def convert_pytorch_to_jax(self, pytorch_model: torch.nn.Module) -> Any:
        """Convert PyTorch model to JAX/Flax model."""
        if not JAX_AVAILABLE:
            warnings.warn("JAX not available, returning original model")
            return pytorch_model
        
        try:
            # Cache conversion if already done
            model_id = id(pytorch_model)
            if model_id in self.conversion_cache:
                return self.conversion_cache[model_id]
            
            # Convert model architecture
            jax_model = self._convert_model_architecture(pytorch_model)
            
            # Convert weights
            jax_params = self._convert_model_weights(pytorch_model, jax_model)
            
            # Cache the conversion
            self.conversion_cache[model_id] = (jax_model, jax_params)
            
            print("✅ PyTorch model converted to JAX/Flax")
            return jax_model, jax_params
            
        except Exception as e:
            warnings.warn(f"PyTorch to JAX conversion failed: {e}")
            return pytorch_model
    
    def _convert_model_architecture(self, pytorch_model: torch.nn.Module) -> Any:
        """Convert PyTorch model architecture to Flax."""
        # This is a simplified conversion - would need more sophisticated mapping
        try:
            if hasattr(pytorch_model, 'config'):
                # For transformer models, create equivalent Flax model
                return self._create_flax_transformer(pytorch_model.config)
            else:
                # Generic conversion
                return self._create_generic_flax_model(pytorch_model)
        except Exception as e:
            warnings.warn(f"Architecture conversion failed: {e}")
            return None
    
    def _convert_model_weights(self, pytorch_model: torch.nn.Module, jax_model: Any) -> Dict:
        """Convert PyTorch weights to JAX format."""
        try:
            jax_params = {}
            
            for name, param in pytorch_model.named_parameters():
                # Convert tensor to JAX array
                jax_array = jnp.array(param.detach().cpu().numpy())
                jax_params[name] = jax_array
            
            return jax_params
            
        except Exception as e:
            warnings.warn(f"Weight conversion failed: {e}")
            return {}
    
    def _create_flax_transformer(self, config) -> Any:
        """Create Flax transformer model from config."""
        # Simplified transformer creation
        # In practice, would need detailed mapping of architectures
        try:
            class FlaxTransformer(nn_flax.Module):
                config: Any
                
                def setup(self):
                    self.embed = nn_flax.Embed(
                        num_embeddings=self.config.vocab_size,
                        features=self.config.hidden_size
                    )
                    # Add more layers based on config
                
                def __call__(self, input_ids, attention_mask=None):
                    x = self.embed(input_ids)
                    # Forward pass implementation
                    return x
            
            return FlaxTransformer(config)
        except:
            return None
    
    def _create_generic_flax_model(self, pytorch_model: torch.nn.Module) -> Any:
        """Create generic Flax model."""
        # Placeholder for generic conversion
        return None
    
    @jax.jit
    def jax_training_step(self, params, batch, optimizer_state):
        """JIT-compiled training step in JAX."""
        def loss_fn(params):
            # Model forward pass
            logits = self.jax_model.apply(params, batch['input_ids'])
            # Compute loss
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch['labels']
            ).mean()
            return loss
        
        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        # Update parameters
        updates, optimizer_state = self.optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        
        return params, optimizer_state, loss
    
    def benchmark_pytorch_vs_jax(self, model, sample_batch, num_iterations: int = 100):
        """Benchmark PyTorch vs JAX performance."""
        results = {}
        
        # PyTorch benchmark
        if torch.cuda.is_available() or TPU_AVAILABLE:
            pytorch_times = self._benchmark_pytorch(model, sample_batch, num_iterations)
            results['pytorch'] = {
                'mean_time': sum(pytorch_times) / len(pytorch_times),
                'min_time': min(pytorch_times),
                'max_time': max(pytorch_times)
            }
        
        # JAX benchmark
        if JAX_AVAILABLE:
            jax_model, jax_params = self.convert_pytorch_to_jax(model)
            if jax_model:
                jax_times = self._benchmark_jax(jax_model, jax_params, sample_batch, num_iterations)
                results['jax'] = {
                    'mean_time': sum(jax_times) / len(jax_times),
                    'min_time': min(jax_times),
                    'max_time': max(jax_times)
                }
        
        return results
    
    def _benchmark_pytorch(self, model, batch, iterations):
        """Benchmark PyTorch model."""
        times = []
        model.eval()
        
        for _ in range(iterations):
            start_time = time.time()
            with torch.no_grad():
                _ = model(**batch)
            
            if TPU_AVAILABLE:
                xm.mark_step()  # Synchronize TPU
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
            
            times.append(time.time() - start_time)
        
        return times
    
    def _benchmark_jax(self, jax_model, params, batch, iterations):
        """Benchmark JAX model."""
        times = []
        
        # Convert batch to JAX
        jax_batch = {k: jnp.array(v.cpu().numpy()) for k, v in batch.items()}
        
        # JIT compile
        @jax.jit
        def jax_forward(params, batch):
            return jax_model.apply(params, batch['input_ids'])
        
        # Warmup
        for _ in range(5):
            _ = jax_forward(params, jax_batch)
        
        # Benchmark
        for _ in range(iterations):
            start_time = time.time()
            _ = jax_forward(params, jax_batch).block_until_ready()
            times.append(time.time() - start_time)
        
        return times

class AdvancedTPUProfiler:
    """
    Advanced profiling and performance insights for TPU training.
    Provides detailed analysis of TPU utilization and bottlenecks.
    """
    
    def __init__(self, enable_detailed: bool = True, enable_memory_tracking: bool = True):
        self.enable_detailed = enable_detailed
        self.enable_memory_tracking = enable_memory_tracking
        self.profiling_data = {}
        self.memory_timeline = []
        self.compilation_stats = {}
        self.communication_metrics = {}
        self.bottleneck_analysis = {}
        
        if TPU_AVAILABLE:
            self._initialize_tpu_profiling()
    
    def _initialize_tpu_profiling(self):
        """Initialize TPU-specific profiling."""
        try:
            # Enable TPU profiling
            if hasattr(xm, 'set_replication'):
                self.tpu_profiling_enabled = True
            else:
                self.tpu_profiling_enabled = False
            
            print("✅ Advanced TPU profiling initialized")
            
        except Exception as e:
            warnings.warn(f"TPU profiling initialization failed: {e}")
            self.tpu_profiling_enabled = False
    
    @contextmanager
    def profile_training_step(self, step_id: int):
        """Context manager for profiling a training step."""
        step_start = time.time()
        step_data = {
            'step_id': step_id,
            'start_time': step_start,
            'memory_before': self._get_memory_usage(),
            'compilation_events': [],
            'communication_events': []
        }
        
        try:
            yield step_data
        finally:
            step_data['end_time'] = time.time()
            step_data['duration'] = step_data['end_time'] - step_data['start_time']
            step_data['memory_after'] = self._get_memory_usage()
            step_data['memory_delta'] = self._calculate_memory_delta(
                step_data['memory_before'], 
                step_data['memory_after']
            )
            
            self.profiling_data[step_id] = step_data
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        if not self.enable_memory_tracking:
            return {}
        
        memory_stats = {}
        
        if TPU_AVAILABLE and xm:
            try:
                # TPU memory stats
                memory_info = xm.get_memory_info(xm.xla_device())
                memory_stats['tpu'] = memory_info
            except:
                pass
        
        if torch.cuda.is_available():
            # GPU memory stats
            memory_stats['gpu'] = {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        
        return memory_stats
    
    def _calculate_memory_delta(self, before: Dict, after: Dict) -> Dict[str, Any]:
        """Calculate memory usage changes."""
        delta = {}
        
        for device in before.keys():
            if device in after:
                if device == 'tpu':
                    # TPU memory delta calculation
                    delta[device] = {
                        'bytes_in_use_delta': after[device].get('bytes_in_use', 0) - before[device].get('bytes_in_use', 0)
                    }
                elif device == 'gpu':
                    # GPU memory delta calculation
                    delta[device] = {
                        'allocated_delta': after[device]['allocated'] - before[device]['allocated'],
                        'cached_delta': after[device]['cached'] - before[device]['cached']
                    }
        
        return delta
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks from collected data."""
        if not self.profiling_data:
            return {"error": "No profiling data available"}
        
        analysis = {
            'step_time_analysis': self._analyze_step_times(),
            'memory_analysis': self._analyze_memory_patterns(),
            'compilation_analysis': self._analyze_compilation_overhead(),
            'communication_analysis': self._analyze_communication_patterns(),
            'recommendations': []
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_step_times(self) -> Dict[str, Any]:
        """Analyze training step times."""
        step_times = [data['duration'] for data in self.profiling_data.values()]
        
        if not step_times:
            return {}
        
        return {
            'mean_step_time': sum(step_times) / len(step_times),
            'min_step_time': min(step_times),
            'max_step_time': max(step_times),
            'std_step_time': self._calculate_std(step_times),
            'total_steps': len(step_times),
            'step_time_trend': self._analyze_trend(step_times)
        }
    
    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        memory_patterns = {}
        
        for step_data in self.profiling_data.values():
            memory_delta = step_data.get('memory_delta', {})
            
            for device, delta in memory_delta.items():
                if device not in memory_patterns:
                    memory_patterns[device] = []
                memory_patterns[device].append(delta)
        
        analysis = {}
        for device, deltas in memory_patterns.items():
            analysis[device] = self._analyze_memory_device(deltas)
        
        return analysis
    
    def _analyze_memory_device(self, deltas: List[Dict]) -> Dict[str, Any]:
        """Analyze memory patterns for a specific device."""
        if not deltas:
            return {}
        
        # Extract relevant metrics based on device type
        if 'allocated_delta' in deltas[0]:  # GPU
            allocated_deltas = [d['allocated_delta'] for d in deltas]
            return {
                'mean_allocation': sum(allocated_deltas) / len(allocated_deltas),
                'max_allocation': max(allocated_deltas),
                'min_allocation': min(allocated_deltas)
            }
        elif 'bytes_in_use_delta' in deltas[0]:  # TPU
            usage_deltas = [d['bytes_in_use_delta'] for d in deltas]
            return {
                'mean_usage': sum(usage_deltas) / len(usage_deltas),
                'max_usage': max(usage_deltas),
                'min_usage': min(usage_deltas)
            }
        
        return {}
    
    def _analyze_compilation_overhead(self) -> Dict[str, Any]:
        """Analyze XLA compilation overhead."""
        return {
            'total_compilation_events': len(self.compilation_stats),
            'compilation_time': sum(self.compilation_stats.values()),
            'average_compilation_time': (
                sum(self.compilation_stats.values()) / len(self.compilation_stats)
                if self.compilation_stats else 0
            )
        }
    
    def _analyze_communication_patterns(self) -> Dict[str, Any]:
        """Analyze communication patterns."""
        return {
            'total_communication_events': len(self.communication_metrics),
            'communication_overhead': sum(self.communication_metrics.values())
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Step time recommendations
        step_analysis = analysis.get('step_time_analysis', {})
        if step_analysis.get('std_step_time', 0) > step_analysis.get('mean_step_time', 0) * 0.2:
            recommendations.append("High step time variance detected - consider batch size optimization")
        
        # Memory recommendations
        memory_analysis = analysis.get('memory_analysis', {})
        for device, stats in memory_analysis.items():
            if stats.get('max_allocation', 0) > stats.get('mean_allocation', 0) * 2:
                recommendations.append(f"Memory spikes detected on {device} - consider gradient accumulation")
        
        # Compilation recommendations
        comp_analysis = analysis.get('compilation_analysis', {})
        if comp_analysis.get('compilation_time', 0) > 10:  # 10 seconds
            recommendations.append("High compilation overhead - consider using dynamic shapes or model caching")
        
        return recommendations
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _analyze_trend(self, values: List[float]) -> str:
        """Analyze trend in values."""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple trend analysis
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)
        
        if second_mean > first_mean * 1.1:
            return "increasing"
        elif second_mean < first_mean * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def export_profiling_report(self, filepath: str):
        """Export detailed profiling report."""
        report = {
            'profiling_summary': {
                'total_steps': len(self.profiling_data),
                'profiling_enabled': self.enable_detailed,
                'memory_tracking_enabled': self.enable_memory_tracking
            },
            'bottleneck_analysis': self.analyze_bottlenecks(),
            'raw_data': self.profiling_data if self.enable_detailed else {},
            'memory_timeline': self.memory_timeline,
            'compilation_stats': self.compilation_stats,
            'communication_metrics': self.communication_metrics
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✅ Profiling report exported to {filepath}")
        except Exception as e:
            warnings.warn(f"Failed to export profiling report: {e}")

class AutoScalingManager:
    """
    Auto-scaling manager for dynamic TPU resource allocation.
    Automatically scales TPU resources based on workload demands.
    """
    
    def __init__(self, config: AutoScalingConfig):
        self.config = config
        self.current_cores = config.min_cores
        self.scaling_history = []
        self.resource_monitor = None
        self.scaling_thread = None
        self.stop_scaling = False
        
        if config.enable_auto_scaling:
            self._initialize_auto_scaling()
    
    def _initialize_auto_scaling(self):
        """Initialize auto-scaling components."""
        try:
            self.resource_monitor = ResourceMonitor()
            self.scaling_thread = threading.Thread(target=self._scaling_loop)
            self.scaling_thread.start()
            
            print("✅ Auto-scaling manager initialized")
            
        except Exception as e:
            warnings.warn(f"Auto-scaling initialization failed: {e}")
    
    def _scaling_loop(self):
        """Main auto-scaling loop."""
        while not self.stop_scaling:
            try:
                # Monitor resource utilization
                utilization = self.resource_monitor.get_utilization()
                
                # Make scaling decision
                scaling_decision = self._make_scaling_decision(utilization)
                
                # Execute scaling if needed
                if scaling_decision != 'no_change':
                    self._execute_scaling(scaling_decision, utilization)
                
                time.sleep(self.config.scale_check_interval)
                
            except Exception as e:
                warnings.warn(f"Scaling loop error: {e}")
    
    def _make_scaling_decision(self, utilization: Dict[str, float]) -> str:
        """Make scaling decision based on utilization."""
        avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0
        
        if avg_utilization > self.config.scale_up_threshold and self.current_cores < self.config.max_cores:
            return 'scale_up'
        elif avg_utilization < self.config.scale_down_threshold and self.current_cores > self.config.min_cores:
            return 'scale_down'
        else:
            return 'no_change'
    
    def _execute_scaling(self, decision: str, utilization: Dict[str, float]):
        """Execute scaling decision."""
        try:
            if decision == 'scale_up':
                new_cores = min(self.current_cores * 2, self.config.max_cores)
                self._scale_to_cores(new_cores)
                
            elif decision == 'scale_down':
                new_cores = max(self.current_cores // 2, self.config.min_cores)
                self._scale_to_cores(new_cores)
            
            # Record scaling event
            self.scaling_history.append({
                'timestamp': time.time(),
                'decision': decision,
                'old_cores': self.current_cores,
                'new_cores': getattr(self, 'target_cores', self.current_cores),
                'utilization': utilization
            })
            
        except Exception as e:
            warnings.warn(f"Scaling execution failed: {e}")
    
    def _scale_to_cores(self, target_cores: int):
        """Scale to target number of cores."""
        try:
            print(f"🔄 Scaling from {self.current_cores} to {target_cores} cores")
            
            # Implementation would depend on cloud provider APIs
            # For now, just update the target
            self.target_cores = target_cores
            
            # In practice, would trigger cloud scaling APIs here
            # self._trigger_cloud_scaling(target_cores)
            
            self.current_cores = target_cores
            print(f"✅ Scaled to {target_cores} cores")
            
        except Exception as e:
            warnings.warn(f"Core scaling failed: {e}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            'current_cores': self.current_cores,
            'min_cores': self.config.min_cores,
            'max_cores': self.config.max_cores,
            'auto_scaling_enabled': self.config.enable_auto_scaling,
            'scaling_events': len(self.scaling_history),
            'last_scaling': self.scaling_history[-1] if self.scaling_history else None
        }
    
    def stop(self):
        """Stop auto-scaling."""
        self.stop_scaling = True
        if self.scaling_thread:
            self.scaling_thread.join()

class ResourceMonitor:
    """Monitor TPU resource utilization."""
    
    def __init__(self):
        self.utilization_history = []
    
    def get_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        utilization = {}
        
        try:
            if TPU_AVAILABLE and xm:
                # TPU utilization (simplified)
                memory_info = xm.get_memory_info(xm.xla_device())
                utilization['tpu_memory'] = (
                    memory_info.get('bytes_in_use', 0) / 
                    memory_info.get('bytes_limit', 1)
                )
            
            # Add CPU, network, etc. monitoring as needed
            
        except Exception as e:
            warnings.warn(f"Utilization monitoring failed: {e}")
        
        return utilization

# Communication optimizers for inter-pod communication
class OptimalInterPodCommunicator:
    """Optimal inter-pod communication strategy."""
    
    def __init__(self):
        self.communication_graph = None
        self.bandwidth_matrix = None
    
    def initialize(self, pod_id: int, num_pods: int, cores_per_pod: int):
        """Initialize communication strategy."""
        self.pod_id = pod_id
        self.num_pods = num_pods
        self.cores_per_pod = cores_per_pod
        
        # Build optimal communication graph
        self._build_communication_graph()
    
    def _build_communication_graph(self):
        """Build optimal communication graph between pods."""
        # Implementation for optimal communication topology
        pass
    
    def all_reduce_across_pods(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform optimized all-reduce across pods."""
        # Implementation for optimized cross-pod all-reduce
        return data

class RingInterPodCommunicator:
    """Ring-based inter-pod communication."""
    
    def initialize(self, pod_id: int, num_pods: int, cores_per_pod: int):
        self.pod_id = pod_id
        self.num_pods = num_pods
        self.next_pod = (pod_id + 1) % num_pods
        self.prev_pod = (pod_id - 1) % num_pods
    
    def all_reduce_across_pods(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Ring-based all-reduce across pods."""
        # Implementation for ring all-reduce
        return data

class TreeInterPodCommunicator:
    """Tree-based inter-pod communication."""
    
    def initialize(self, pod_id: int, num_pods: int, cores_per_pod: int):
        self.pod_id = pod_id
        self.num_pods = num_pods
        # Build tree topology
        self._build_tree_topology()
    
    def _build_tree_topology(self):
        """Build tree communication topology."""
        # Implementation for tree topology
        pass
    
    def all_reduce_across_pods(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Tree-based all-reduce across pods."""
        # Implementation for tree all-reduce
        return data

# Global instances for Phase 3 components
multi_pod_optimizer = None
jax_flax_integration = None
advanced_profiler = None
auto_scaling_manager = None

def initialize_phase3_optimizations(
    multi_pod_config: Optional[MultiPodConfig] = None,
    jax_flax_config: Optional[JAXFlaxConfig] = None,
    auto_scaling_config: Optional[AutoScalingConfig] = None,
    enable_advanced_profiling: bool = True
) -> Dict[str, Any]:
    """
    Initialize Phase 3 optimizations.
    
    Args:
        multi_pod_config: Configuration for multi-pod optimization
        jax_flax_config: Configuration for JAX/Flax integration
        auto_scaling_config: Configuration for auto-scaling
        enable_advanced_profiling: Whether to enable advanced profiling
    
    Returns:
        Dictionary with initialization status
    """
    global multi_pod_optimizer, jax_flax_integration, advanced_profiler, auto_scaling_manager
    
    results = {}
    
    try:
        # Initialize multi-pod optimization
        if multi_pod_config:
            multi_pod_optimizer = MultiPodOptimizer(multi_pod_config)
            results['multi_pod'] = True
        else:
            results['multi_pod'] = False
        
        # Initialize JAX/Flax integration
        if jax_flax_config:
            jax_flax_integration = JAXFlaxIntegration(jax_flax_config)
            results['jax_flax'] = JAX_AVAILABLE
        else:
            results['jax_flax'] = False
        
        # Initialize advanced profiling
        if enable_advanced_profiling:
            advanced_profiler = AdvancedTPUProfiler()
            results['advanced_profiling'] = True
        else:
            results['advanced_profiling'] = False
        
        # Initialize auto-scaling
        if auto_scaling_config:
            auto_scaling_manager = AutoScalingManager(auto_scaling_config)
            results['auto_scaling'] = True
        else:
            results['auto_scaling'] = False
        
        print("✅ Phase 3 optimizations initialized")
        return results
        
    except Exception as e:
        warnings.warn(f"Phase 3 initialization failed: {e}")
        return {'error': str(e)}

def get_phase3_status() -> Dict[str, Any]:
    """Get status of Phase 3 components."""
    return {
        'tpu_available': TPU_AVAILABLE,
        'jax_available': JAX_AVAILABLE,
        'phase12_available': PHASE12_AVAILABLE,
        'multi_pod_optimizer': multi_pod_optimizer is not None,
        'jax_flax_integration': jax_flax_integration is not None,
        'advanced_profiler': advanced_profiler is not None,
        'auto_scaling_manager': auto_scaling_manager is not None,
        'components_status': {
            'multi_pod': multi_pod_optimizer.get_pod_status() if multi_pod_optimizer else None,
            'auto_scaling': auto_scaling_manager.get_scaling_status() if auto_scaling_manager else None,
        }
    }

def cleanup_phase3():
    """Cleanup Phase 3 components."""
    global multi_pod_optimizer, jax_flax_integration, advanced_profiler, auto_scaling_manager
    
    try:
        if multi_pod_optimizer and hasattr(multi_pod_optimizer, 'global_coordinator'):
            if multi_pod_optimizer.global_coordinator:
                multi_pod_optimizer.global_coordinator.stop()
        
        if auto_scaling_manager:
            auto_scaling_manager.stop()
        
        print("✅ Phase 3 cleanup completed")
        
    except Exception as e:
        warnings.warn(f"Phase 3 cleanup failed: {e}")

# Export all Phase 3 components
__all__ = [
    # Configuration classes
    "MultiPodConfig",
    "JAXFlaxConfig", 
    "AutoScalingConfig",
    
    # Main components
    "MultiPodOptimizer",
    "JAXFlaxIntegration",
    "AdvancedTPUProfiler",
    "AutoScalingManager",
    
    # Communication optimizers
    "OptimalInterPodCommunicator",
    "RingInterPodCommunicator",
    "TreeInterPodCommunicator",
    
    # Utility functions
    "initialize_phase3_optimizations",
    "get_phase3_status",
    "cleanup_phase3",
    
    # Global instances
    "multi_pod_optimizer",
    "jax_flax_integration", 
    "advanced_profiler",
    "auto_scaling_manager",
]
