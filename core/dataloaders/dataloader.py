"""
Data Loader Module
"""
from typing import Callable, Optional
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from core.configurations.base import BaseConfiguration
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, token_ids, config: BaseConfiguration):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - config.model_max_sequence, config.strides):
            input_chunk = token_ids[i: i+config.model_max_sequence]
            target_chunk = token_ids[i + 1 : i + config.model_max_sequence + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def load_tokenizer(tokenizer_path: str):
    sp_model = spm.SentencePieceProcessor(model_file=tokenizer_path)
    return sp_model

def tokenize_text(tokenizer, text):
    """Tokenize text using the provided tokenizer."""
    return tokenizer.encode(text)

def dataloader_v1(dataset, tokenizer, config: BaseConfiguration):
    # Tokenize the text dataset
    token_ids = []
    for index, sample in enumerate(dataset):
        token_ids.extend(tokenize_text(tokenizer, sample['content']))

    # Create dataset
    dataset = CustomDataset(token_ids, config)


    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=config.dataset_batch_size,
        shuffle=config.dataset_shuffle,
        num_workers=4
    )

    return data_loader


def create_data_loader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None
) -> DataLoader:
    """
    Create a PyTorch DataLoader with optimized settings.

    Args:
        dataset: PyTorch Dataset object
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of subprocesses for data loading
        pin_memory: Whether to pin memory in GPU training
        collate_fn: Custom collate function for batching
        drop_last: Whether to drop the last incomplete batch
        generator: Random number generator for reproducibility

    Returns:
        DataLoader: Configured PyTorch DataLoader
    """


    if collate_fn is None:
        raise ValueError("collator function not provided")

    # Choose sampler based on shuffle parameter
    if shuffle:
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    # Create DataLoader with optimized settings
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        # Worker init function for reproducibility
        worker_init_fn=lambda worker_id: torch.manual_seed(torch.initial_seed() + worker_id)
    )

    return loader