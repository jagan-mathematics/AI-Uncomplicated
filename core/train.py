"""Training implementation"""
import sys

sys.path.append("/workspace/vipin_g6/personal/pretraining/AI-Uncomplicated/")
from tqdm import tqdm
import pickle
from pathlib import Path
import torch
import wandb
from datasets import load_dataset
from typing import Optional, Dict, Any

from core.configurations.base import BaseConfiguration
from core.dataloaders.dataloader import dataloader_v1, load_tokenizer
from core.models.model import LLM

run = wandb.init()# Init wandb

def loss_batch_calculation(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)['logits']
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
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


def train_model(model: LLM, train_loader, val_loader, optimizer, device, config: BaseConfiguration):
    train_losses, val_losses = [], []
    global_steps = -1

    for epoch in tqdm(range(config.num_epochs), desc="Training:"):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_batch_calculation(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_steps += 1

            # Need to implement evaluation also
            if global_steps % config.eval_frequency == 0:
                train_loss, eval_loss = evaluate_model(model, train_loader, val_loader, device, config.eval_iter)
                train_losses.append(train_loss)
                val_losses.append(eval_loss)
                print(f"Ep {epoch + 1} (Step {global_steps:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {eval_loss:.3f}")
                run.log(
                    {
                        "epoch" : epoch + 1,
                        "steps" : global_steps,
                        "train_loss" : train_loss,
                        "val_loss" : eval_loss
                    }
                )

    return train_losses, val_losses


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


def main(config: Dict[str, Any]):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_loader = CachedDatasetLoader()
    train_dataset = dataset_loader.load_with_cache("TigerResearch/pretrain_en", range_start=0, range_end=10000)

    validation_dataset = dataset_loader.load_with_cache("TigerResearch/pretrain_en", range_start=10001, range_end=12000)

    tokenizer = load_tokenizer(config.tokenizer_path)
    train_loader = dataloader_v1(train_dataset, tokenizer, config)
    val_loader = dataloader_v1(validation_dataset, tokenizer, config)

    # Initialize the model
    model = LLM(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_loss, val_loss = train_model(model, train_loader, val_loader, optimizer, device, config)

    return train_loss, val_loss, model


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    config = BaseConfiguration(model_name="small_lm", num_layers=2, hidden_dim=128, intermediate_dim=512,
                               max_positions=256, vocabulary_size=64000, num_heads=2, attention_dropout=0.05,
                               batch_size=8, weight_decay=0.01,
                               learning_rate=5e-4,
                               tokenizer_path="/workspace/vipin_g6/personal/pretraining/english_tokenizer/english_tokenizer.model",
                               dataset_batch_size=16, dataset_shuffle=True, num_epochs=2, eval_frequency=1,
                               eval_iter=10,
                               model_max_sequence=256)

    print(f"====config present {config}=======")
    _, _, model = main(config)
    print(f"======= Model Trained Successfully ==========")
    save_object(model, "pretrained_model/")
    print(f"======= Model Saved Successfully ==========")
