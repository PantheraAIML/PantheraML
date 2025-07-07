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
    EXPERIMENTAL: TPU-enabled trainer for PantheraML.
    
    Warning: This is experimental TPU support and may have limitations.
    """
    
    def __init__(self, tpu_config: Optional[MultiTPUConfig] = None, *args, **kwargs):
        if not TPU_AVAILABLE:
            raise RuntimeError(
                "🧪 EXPERIMENTAL: TPU support requires torch_xla. "
                "Install with: pip install torch_xla"
            )
        
        print("🧪 EXPERIMENTAL: Initializing TPU trainer...")
        print("   ⚠️  Note: TPU support is experimental and may have limitations")
        
        self.tpu_config = tpu_config or MultiTPUConfig()
        super().__init__(*args, **kwargs)
        
        # Move model to TPU device
        try:
            self.model = self.model.to(get_tpu_device())
            print(f"   ✅ Model moved to TPU device")
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to move model to TPU: {e}")
    
    def training_step(self, model, inputs):
        """Override training step for TPU-specific optimizations."""
        try:
            # Standard training step
            result = super().training_step(model, inputs)
            
            # TPU-specific synchronization
            synchronize_tpu()
            
            return result
        except Exception as e:
            print(f"🧪 EXPERIMENTAL TPU: Training step error: {e}")
            raise
    
    def get_model_memory_usage(self):
        """Get TPU memory usage (experimental)."""
        if not TPU_AVAILABLE:
            return {}
        
        try:
            # TPU memory tracking is limited
            return {
                'tpu_core': {
                    'rank': get_tpu_rank(),
                    'world_size': get_tpu_world_size(),
                    'device': str(get_tpu_device()),
                }
            }
        except Exception as e:
            print(f"🧪 EXPERIMENTAL TPU: Memory tracking error: {e}")
            return {}
    
    def print_memory_stats(self):
        """Print TPU memory usage."""
        if is_tpu_main_process():
            memory_stats = self.get_model_memory_usage()
            print("\n🧪 EXPERIMENTAL TPU Status:")
            for device_id, stats in memory_stats.items():
                print(f"   {device_id.upper()}: Rank {stats['rank']}/{stats['world_size']}, "
                      f"Device: {stats['device']}")
            print()
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Override save model for TPU-specific handling."""
        print("🧪 EXPERIMENTAL: Saving TPU model...")
        
        # Synchronize before saving
        synchronize_tpu()
        
        # Only save on main process
        if is_tpu_main_process():
            super().save_model(output_dir, _internal_call)
        else:
            print("   Skipping save on non-main TPU process")
