"""
Available autowrap policy from torch 2.4.0
__all__ = [
    "always_wrap_policy",
    "lambda_auto_wrap_policy",
    "transformer_auto_wrap_policy",
    "size_based_auto_wrap_policy",
    "enable_wrap",
    "wrap",
    "CustomPolicy",
    "ModuleWrapPolicy",
]
"""

from core.models.GI_01.main.config import ModelConfig, DatasetConfig, TrainingConfig
from core.dataloaders.dataloader import load_tokenizer


from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any


# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp.fully_sharded_data_parallel import (
#     ShardingStrategy,
#     MixedPrecision
# )
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


## Initialize configurations
model_config = ModelConfig(model_name="Construe",
                           num_layers = 2,
                            padding_id = 0,
                            hidden_dim = 512,
                            intermediate_dim = 3072,
                            max_positions = 2048,
                            layer_norm_eps = 1e-05,
                            model_max_sequence = 2048,
                            num_heads = 8,
                            attention_dropout = 0.1)

dataset_config = DatasetConfig(dataset_path="./dataset",
                               dataset_shuffle=True)
training_config = TrainingConfig(tokenizer_path="/root/AI-Uncomplicated/core/tokenizer/bpe/pre_trained/europian_ml")

## Load model and tokenizer

tokenizer =  load_tokenizer(training_config.tokenizer_path)

