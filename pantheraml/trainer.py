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

import warnings
from dataclasses import dataclass, field
from typing import Optional
from functools import wraps

import trl
import inspect
from trl import SFTTrainer
from . import is_bfloat16_supported
from unsloth_zoo.training_utils import (
    unsloth_train as _unsloth_train,
)
from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator,
)
from packaging.version import Version
import dataclasses

# Alias for backward compatibility and consistent naming
PantheraMLVisionDataCollator = UnslothVisionDataCollator

# Multi-GPU support
import torch
import torch.distributed as dist
from .distributed import (
    MultiGPUConfig,
    MultiTPUConfig,  # Experimental TPU config
    setup_multi_gpu,
    setup_multi_tpu,  # Experimental TPU setup
    cleanup_multi_gpu,
    cleanup_multi_tpu,  # Experimental TPU cleanup
    is_distributed_available,
    is_tpu_available,  # Experimental TPU check
    get_rank,
    get_world_size,
    is_main_process,
    get_tpu_rank,  # Experimental TPU rank
    get_tpu_world_size,  # Experimental TPU world size
    is_tpu_main_process,  # Experimental TPU main process
    wrap_model_for_distributed,
    get_sampler_for_distributed,
    all_reduce_scalar,
    barrier,
    get_tpu_device,  # Experimental TPU device
    synchronize_tpu,  # Experimental TPU sync
)

# Try to import TPU support
try:
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

# Phase 2 TPU Performance Integration
try:
    from .kernels.tpu_performance import (
        XLAAttentionOptimizer, ModelShardManager, DynamicShapeManager,
        TPUCommunicationOptimizer, TPUPerformanceProfiler
    )
    PHASE2_TPU_AVAILABLE = True
except ImportError:
    PHASE2_TPU_AVAILABLE = False

__all__ = [
    "PantheraMLTrainingArguments",
    "PantheraMLTrainer", 
    "PantheraMLDistributedTrainer",
    "PantheraMLTPUTrainer",  # Experimental TPU trainer
    "pantheraml_train",
    "PantheraMLVisionDataCollator",
    "MultiGPUConfig",
    "MultiTPUConfig",  # Experimental TPU config
]

# PantheraML gradient accumulation fix:
from transformers import __version__ as transformers_version
if Version(transformers_version) > Version("4.45.2"):
    def pantheraml_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)
    pass
else:
    def pantheraml_train(trainer, *args, **kwargs):
        if len(args) != 0 or len(kwargs) != 0:
            raise RuntimeError(
                "PantheraML: Our custom gradient accumulation fixed trainer does not support other arguments.\n"\
                "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
                '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`'
            )
        print(
            "PantheraML: Using our custom gradient accumulation fixed trainer, which is not feature complete.\n"\
            "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
            '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`'
        )
        return _unsloth_train(trainer)
    pass
pass

try:
    from trl import SFTConfig as TrainingArguments
except:
    from transformers import TrainingArguments
pass

class PantheraMLTrainingArguments(TrainingArguments):
    def __init__(self, embedding_learning_rate: float = None, *args, **kwargs):
        embedding_learning_rate = embedding_learning_rate
        super().__init__(*args, **kwargs)
pass


def _create_pantheraml_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = \
    {
        "non_embeddings" : {},
        "embeddings"     : {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[:-len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".")+1:]
            print(f"PantheraML: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}.")
            param_groups["embeddings"]    [name] = param
        else:
            param_groups["non_embeddings"][name] = param
        pass
    pass

    optimizer_grouped_parameters = [
        {
            "params"       : list(param_groups["non_embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : lr,
        },
        {
            "params"       : list(param_groups["embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
pass


class PantheraMLTrainer(SFTTrainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None: return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_pantheraml_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        pass
        return self.optimizer
    pass
pass


class PantheraMLDistributedTrainer(PantheraMLTrainer):
    """
    Enhanced PantheraMLTrainer with built-in multi-GPU support.
    
    This trainer automatically handles:
    - Multi-GPU model distribution
    - Distributed data loading
    - Gradient synchronization
    - Performance monitoring across GPUs
    """
    
    def __init__(
        self,
        multi_gpu_config: Optional[MultiGPUConfig] = None,
        auto_setup_distributed: bool = True,
        *args,
        **kwargs
    ):
        # Setup distributed training if requested
        self.multi_gpu_config = multi_gpu_config or MultiGPUConfig()
        self.auto_setup_distributed = auto_setup_distributed
        
        if auto_setup_distributed and is_distributed_available():
            setup_multi_gpu(self.multi_gpu_config)
        
        super().__init__(*args, **kwargs)
        
        # Wrap model for distributed training
        if hasattr(self, 'model') and self.model is not None:
            self.model = wrap_model_for_distributed(
                self.model, 
                self.multi_gpu_config
            )
    
    def get_train_dataloader(self):
        """Override to use distributed sampler when appropriate."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        # Get distributed sampler if needed
        train_sampler = get_sampler_for_distributed(
            train_dataset, 
            shuffle=True
        )
        
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use distributed sampler for evaluation."""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # Get distributed sampler for evaluation (no shuffling)
        eval_sampler = get_sampler_for_distributed(
            eval_dataset,
            shuffle=False
        )
        
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=eval_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override to handle distributed loss computation."""
        loss = super().compute_loss(model, inputs, return_outputs)
        
        # For distributed training, we might want to sync losses
        if get_world_size() > 1:
            if return_outputs:
                loss_value, outputs = loss
                # Average loss across all GPUs
                loss_value = all_reduce_scalar(loss_value.item(), average=True)
                return torch.tensor(loss_value, device=inputs.get('input_ids', inputs.get('pixel_values')).device), outputs
            else:
                # Average loss across all GPUs
                loss_value = all_reduce_scalar(loss.item(), average=True)
                return torch.tensor(loss_value, device=inputs.get('input_ids', inputs.get('pixel_values')).device)
        
        return loss
    
    def log(self, logs: dict = None):
        """Override to ensure logging only happens on main process."""
        if is_main_process():
            super().log(logs)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Override to save model only on main process."""
        if is_main_process():
            super().save_model(output_dir, _internal_call)
        
        # Ensure all processes wait for the main process to finish saving
        barrier()
    
    def evaluation_loop(self, *args, **kwargs):
        """Override to handle distributed evaluation."""
        # Synchronize before evaluation
        barrier()
        
        result = super().evaluation_loop(*args, **kwargs)
        
        # Synchronize after evaluation
        barrier()
        
        return result
    
    def train(self, *args, **kwargs):
        """Override to add distributed training setup and cleanup."""
        if is_main_process():
            print(f"🚀 Starting distributed training on {get_world_size()} GPUs")
            print(f"   Current process rank: {get_rank()}")
            print(f"   Multi-GPU config: {self.multi_gpu_config.__dict__}")
        
        # Synchronize all processes before starting training
        barrier()
        
        try:
            result = super().train(*args, **kwargs)
            
            if is_main_process():
                print("✅ Distributed training completed successfully")
            
            return result
        
        except Exception as e:
            if is_main_process():
                print(f"❌ Error during distributed training: {e}")
            raise
        
        finally:
            # Cleanup distributed environment if we set it up
            if self.auto_setup_distributed:
                cleanup_multi_gpu()
    
    def create_optimizer(self):
        """Override to handle optimizer creation in distributed setting."""
        optimizer = super().create_optimizer()
        
        # For distributed training, we might want to scale learning rates
        if get_world_size() > 1 and hasattr(self.args, 'auto_scale_lr') and self.args.auto_scale_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= get_world_size()
                if is_main_process():
                    print(f"Scaled learning rate by world size: {param_group['lr']}")
        
        return optimizer
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Override to handle checkpointing in distributed setting."""
        if is_main_process():
            super()._save_checkpoint(model, trial, metrics)
        
        # Wait for main process to finish checkpointing
        barrier()
    
    def get_model_memory_usage(self):
        """Get memory usage statistics across all GPUs."""
        if not torch.cuda.is_available():
            return {}
        
        memory_stats = {}
        for i in range(torch.cuda.device_count()):
            memory_stats[f'gpu_{i}'] = {
                'allocated': torch.cuda.memory_allocated(i) / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved(i) / 1024**3,  # GB
                'max_allocated': torch.cuda.max_memory_allocated(i) / 1024**3,  # GB
            }
        
        return memory_stats
    
    def print_memory_stats(self):
        """Print memory usage across all GPUs."""
        if is_main_process():
            memory_stats = self.get_model_memory_usage()
            print("\n📊 GPU Memory Usage:")
            for gpu_id, stats in memory_stats.items():
                print(f"   {gpu_id.upper()}: {stats['allocated']:.2f}GB allocated, "
                      f"{stats['cached']:.2f}GB cached, "
                      f"{stats['max_allocated']:.2f}GB peak")
            print()

class PantheraMLTPUTrainer(PantheraMLTrainer):
    """
    Enhanced TPU trainer with Phase 1 (error/memory) and Phase 2 (performance) support.
    
    Features:
    - Phase 1: Error handling, memory management, XLA integration
    - Phase 2: Performance optimization, model sharding, dynamic shapes
    """
    
    def __init__(self, *args, **kwargs):
        # Extract TPU-specific config
        self.tpu_config = kwargs.pop('tpu_config', {})
        self.enable_phase2 = kwargs.pop('enable_phase2', True)
        
        super().__init__(*args, **kwargs)
        
        # Initialize Phase 1 components
        self.tpu_error_handler = tpu_error_handler if TPU_AVAILABLE else None
        self.tpu_memory = tpu_memory_manager if TPU_AVAILABLE else None
        self.xla_optimizer = xla_optimizer if TPU_AVAILABLE else None
        self.tpu_config_manager = tpu_config_manager if TPU_AVAILABLE else None
        
        # Initialize Phase 2 components if available and enabled
        self.phase2_enabled = PHASE2_TPU_AVAILABLE and self.enable_phase2 and TPU_AVAILABLE
        if self.phase2_enabled:
            self._init_phase2_components()
        
        # Setup TPU configuration
        if TPU_AVAILABLE and self.tpu_config_manager:
            self.tpu_config_manager.setup_tpu_config(self.tpu_config)
    
    def _init_phase2_components(self):
        """Initialize Phase 2 performance components."""
        try:
            # XLA Attention Optimizer
            self.xla_attention = XLAAttentionOptimizer(
                use_flash_attention=self.tpu_config.get('use_flash_attention', True),
                use_memory_efficient=self.tpu_config.get('use_memory_efficient', True)
            )
            
            # Model Sharding Manager
            self.shard_manager = ModelShardManager(
                num_shards=self.tpu_config.get('num_shards', 8),
                shard_axis=self.tpu_config.get('shard_axis', 0)
            )
            
            # Dynamic Shape Manager
            self.shape_manager = DynamicShapeManager(
                max_length=self.tpu_config.get('max_length', 2048),
                bucket_size=self.tpu_config.get('bucket_size', 64)
            )
            
            # Communication Optimizer
            self.comm_optimizer = TPUCommunicationOptimizer()
            
            # Performance Profiler
            self.profiler = TPUPerformanceProfiler(
                enable_detailed=self.tpu_config.get('enable_profiling', False)
            )
            
            print("✅ Phase 2 TPU performance components initialized")
            
        except Exception as e:
            print(f"⚠️ Phase 2 initialization failed: {e}")
            self.phase2_enabled = False
    
    def training_step(self, model, inputs):
        """Enhanced training step with Phase 2 optimizations."""
        if not TPU_AVAILABLE:
            return super().training_step(model, inputs)
        
        try:
            # Phase 1: Memory management
            if self.tpu_memory:
                self.tpu_memory.clear_cache()
            
            # Phase 2: Dynamic shape optimization
            if self.phase2_enabled and hasattr(self, 'shape_manager'):
                inputs = self.shape_manager.optimize_batch(inputs)
            
            # Phase 2: Performance profiling start
            if self.phase2_enabled and hasattr(self, 'profiler'):
                self.profiler.start_step()
            
            # Standard training step with XLA optimization
            if self.xla_optimizer:
                with self.xla_optimizer.optimize_step():
                    loss = super().training_step(model, inputs)
            else:
                loss = super().training_step(model, inputs)
            
            # Phase 2: Performance profiling end
            if self.phase2_enabled and hasattr(self, 'profiler'):
                self.profiler.end_step()
            
            return loss
            
        except Exception as e:
            if self.tpu_error_handler:
                return self.tpu_error_handler.handle_training_error(e, model, inputs)
            else:
                raise e
    
    def _prepare_model_for_training(self, model):
        """Prepare model with Phase 2 optimizations."""
        model = super()._prepare_model_for_training(model)
        
        if not self.phase2_enabled:
            return model
        
        try:
            # Phase 2: Model sharding
            if hasattr(self, 'shard_manager'):
                model = self.shard_manager.shard_model(model)
                print(f"✅ Model sharded across {self.shard_manager.num_shards} shards")
            
            # Phase 2: XLA attention optimization
            if hasattr(self, 'xla_attention'):
                model = self.xla_attention.optimize_model(model)
                print("✅ XLA attention optimizations applied")
            
            return model
            
        except Exception as e:
            print(f"⚠️ Phase 2 model preparation failed: {e}")
            return model
    
    def get_train_dataloader(self):
        """Enhanced dataloader with Phase 2 optimizations."""
        dataloader = super().get_train_dataloader()
        
        if not self.phase2_enabled or not hasattr(self, 'shape_manager'):
            return dataloader
        
        try:
            # Phase 2: Dynamic shape optimization for dataloader
            dataloader = self.shape_manager.optimize_dataloader(dataloader)
            print("✅ Dataloader optimized for dynamic shapes")
            return dataloader
            
        except Exception as e:
            print(f"⚠️ Dataloader optimization failed: {e}")
            return dataloader
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Enhanced checkpoint saving with TPU synchronization."""
        if TPU_AVAILABLE and xm:
            # Synchronize before saving
            xm.mark_step()
            if self.is_world_process_zero():
                print("🔄 Synchronizing TPU devices before checkpoint...")
        
        # Phase 2: Communication optimization for checkpointing
        if self.phase2_enabled and hasattr(self, 'comm_optimizer'):
            with self.comm_optimizer.optimize_checkpoint():
                return super()._save_checkpoint(model, trial, metrics)
        else:
            return super()._save_checkpoint(model, trial, metrics)
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Enhanced prediction step with Phase 2 optimizations."""
        if not TPU_AVAILABLE:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        try:
            # Phase 2: Dynamic shape optimization for inference
            if self.phase2_enabled and hasattr(self, 'shape_manager'):
                inputs = self.shape_manager.optimize_inference_batch(inputs)
            
            # Standard prediction with XLA optimization
            if self.xla_optimizer:
                with self.xla_optimizer.optimize_step():
                    return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            else:
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
                
        except Exception as e:
            if self.tpu_error_handler:
                return self.tpu_error_handler.handle_inference_error(e, model, inputs)
            else:
                raise e
    
    def get_performance_metrics(self):
        """Get Phase 2 performance metrics."""
        metrics = {}
        
        if self.phase2_enabled and hasattr(self, 'profiler'):
            metrics.update(self.profiler.get_metrics())
        
        if self.phase2_enabled and hasattr(self, 'comm_optimizer'):
            metrics.update(self.comm_optimizer.get_communication_stats())
        
        if self.tpu_memory:
            metrics.update(self.tpu_memory.get_memory_stats())
        
        return metrics
    
    def cleanup(self):
        """Enhanced cleanup with Phase 2 components."""
        try:
            # Phase 2 cleanup
            if self.phase2_enabled:
                if hasattr(self, 'profiler'):
                    self.profiler.finalize()
                if hasattr(self, 'comm_optimizer'):
                    self.comm_optimizer.cleanup()
                if hasattr(self, 'shard_manager'):
                    self.shard_manager.cleanup()
            
            # Phase 1 cleanup
            if self.tpu_memory:
                self.tpu_memory.cleanup()
            
            print("✅ TPU trainer cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
