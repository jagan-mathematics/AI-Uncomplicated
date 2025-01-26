"""Full FSDP Training Implementation"""
import sys

sys.path.append("/workspace/AI-Uncomplicated/")
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from tqdm import tqdm
from pathlib import Path
import pickle
from typing import Optional, Dict, Any

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
    MixedPrecision
)

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
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


from transformers import (
    AutoConfig, 
    AutoModelForCausalLM
)
from datasets import load_dataset

from core.configurations.base import BaseConfiguration
from core.dataloaders.dataloader import dataloader_v1, load_tokenizer
from core.models.model import LLM

class CachedDatasetLoader:
    def __init__(self, cache_dir: str = ".cache/datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, dataset_name: str, subset_range: tuple) -> Path:
        range_str = f"{subset_range[0]}-{subset_range[1]}"
        return self.cache_dir / f"{dataset_name.replace('/', '_')}_{range_str}.pkl"

    def load_with_cache(self, dataset_name: str, split: str = "train", range_start: int = 0,
                        range_end: Optional[int] = None, force_reload: bool = False) -> Any:
        subset_range = (range_start, range_end if range_end is not None else "end")
        cache_path = self._get_cache_path(dataset_name, subset_range)

        if not force_reload and cache_path.exists():
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"Loading dataset from source: {dataset_name}")
        dataset = load_dataset(dataset_name)[split]

        if range_end is not None:
            dataset = dataset.select(range(range_start, range_end))
        else:
            dataset = dataset.select(range(range_start, len(dataset)))

        print(f"Caching dataset to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

        return dataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("cuda", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def loss_batch_calculation(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)['logits']
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    
    num_batches = num_batches or len(data_loader)
    num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = loss_batch_calculation(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_model(rank, model, train_loader, val_loader, optimizer, config: BaseConfiguration):
    run = None
    if rank == 0:
        run = wandb.init(project="fsdp-training")

    train_losses, val_losses = [], []
    global_steps = -1

    for epoch in tqdm(range(config.num_epochs), desc=f"Training on Rank {rank}"):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_batch_calculation(input_batch, target_batch, model, torch.device(f'cuda:{rank}'))
            loss.backward()
            optimizer.step()
            global_steps += 1


            if rank == 0 and global_steps % config.eval_frequency == 0:
                train_loss, eval_loss = evaluate_model(
                    model, train_loader, val_loader, 
                    torch.device(f'cuda:{rank}'), config.eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(eval_loss)
                
                print(f"Ep {epoch + 1} (Step {global_steps:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {eval_loss:.3f}")
                
                if run:
                    run.log({
                        "epoch": epoch + 1,
                        "steps": global_steps,
                        "train_loss": train_loss,
                        "val_loss": eval_loss
                    })

        if rank == 0:
            save_fsdp_checkpoint(model, 'fsdp_checkpoints', epoch)

    return train_losses, val_losses

def save_fsdp_checkpoint(model, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    
    if dist.get_rank() == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch
        }, os.path.join(save_dir, f'model_epoch_{epoch}.pt'))

def consolidate_fsdp_model(
    fsdp_checkpoint_path: str,
    consolidated_model_path: str,
    model_config: Dict[str, Any] = None
):
    os.makedirs(consolidated_model_path, exist_ok=True)

    if model_config is None:
        model_config = {
            'model_type': 'zohoLM',
            'hidden_size': 128,
            'num_attention_heads': 2,
            'num_hidden_layers': 2,
            'vocab_size': 64000
        }

    config = AutoConfig.for_model(**model_config)
    config.save_pretrained(consolidated_model_path)

    model = AutoModelForCausalLM.from_config(config)

    checkpoint_files = [
        f for f in os.listdir(fsdp_checkpoint_path) 
        if f.startswith("model_") and f.endswith(".pt")
    ]
    
    if not checkpoint_files:
        raise ValueError("No model checkpoint files found")

    consolidated_state_dict = {}
    for file in checkpoint_files:
        checkpoint = torch.load(
            os.path.join(fsdp_checkpoint_path, file), 
            map_location='cpu'
        )
        
        for key, value in checkpoint.get('model_state_dict', {}).items():
            clean_key = key.replace('_fsdp_wrapped_module.', '')
            consolidated_state_dict[clean_key] = value

    model.load_state_dict(consolidated_state_dict)

    model.save_pretrained(consolidated_model_path)
    print(f"Consolidated model saved to {consolidated_model_path}")

def main(rank, world_size, config: Dict[str, Any]):
    """Main training routine"""
    setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')
    torch.manual_seed(42 + rank)

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    )

    dataset_loader = CachedDatasetLoader()
    train_dataset = dataset_loader.load_with_cache(
        "TigerResearch/pretrain_en", 
        range_start=0, 
        range_end=10000
    )
    validation_dataset = dataset_loader.load_with_cache(
        "TigerResearch/pretrain_en", 
        range_start=10001, 
        range_end=12000
    )

    tokenizer = load_tokenizer(config.tokenizer_path)
    train_loader = dataloader_v1(train_dataset, tokenizer, config)
    val_loader = dataloader_v1(validation_dataset, tokenizer, config)

    model = LLM(config)
    model = FSDP(
        model, 
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=transformer_auto_wrap_policy,
        mixed_precision=mixed_precision_policy
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )

    train_loss, val_loss = train_model(rank, model, train_loader, val_loader, optimizer, config)

    if rank == 0:
        consolidate_fsdp_model(
            fsdp_checkpoint_path='fsdp_checkpoints', 
            consolidated_model_path='hf_model_checkpoints'
        )

    cleanup()
    return train_loss, val_loss, model

def run_training(config):
    devices = [0, 2]
    world_size = len(devices)

    mp.spawn(
        main, 
        args=(world_size, config),
        nprocs=world_size
    )

if __name__ == "__main__":
    config = BaseConfiguration(
        model_name="small_lm", 
        num_layers=2, 
        hidden_dim=128, 
        intermediate_dim=512,
        max_positions=256, 
        vocabulary_size=64000, 
        num_heads=2, 
        attention_dropout=0.05,
        batch_size=8, 
        weight_decay=0.01,
        learning_rate=5e-4,
        tokenizer_path="/workspace/AI-Uncomplicated/tokenizer/english_tokenizer.model",
        dataset_batch_size=16, 
        dataset_shuffle=True, 
        num_epochs=2, 
        eval_frequency=1,
        eval_iter=10,
        model_max_sequence=256
    )

    print(f"====Training Configuration: {config}=======")
    run_training(config)
    print("======= Model Training and Saving Completed ==========")
    # TODO: need to save tokenizer as HF Tokenizer 