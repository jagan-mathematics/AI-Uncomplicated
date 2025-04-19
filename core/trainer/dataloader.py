"""
Data Loader Module
"""
import os
from pathlib import Path

import numpy as np
from dataclasses import field, dataclass
from typing import Callable, Optional, Dict, Any, TypedDict
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from core.configurations.base import BaseConfiguration
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader

from core.tokenizer.tokenizer_loader import TokenizerArgs


class JSONLState(TypedDict):
    """Represents the current state of a JSON line reader.

    Attributes:
        content (Dict): The JSON content of the line.
        file_path (str): The path to the JSONL file.
        position (int): The file position after reading the line (in bytes).
        window (int): The window size used for iteration.
        offset (int): The offset used for iteration.
        current_iter (Optional[int]): Number of iterations over the jsonl file (for infinite iteration).
    """

    file_path: str
    position: int
    block_size: int
    offset: int
    current_iter: int


class PrefetchState(TypedDict):
    """Represents the current state of a prefetching iterator.

    Attributes:
        prefetch_buffer: numpy array to store prefetched data
        seq_idx: int index of the current sequence to resume from
        rng_state: dict numpy bit generator state used to resume rng
    """

    it_state: Any
    seq_idx: int
    rng_state: Dict[str, Any]
    prefetch_size: int
    batch_size: int


class MultiChoiceState(TypedDict):
    """Represents the current state of a Multi choice iterator.

    Attributes:
        root_dir: path to dataset root directory
        sources Dict[str, float]: Dict from subdirectory to the weight used for sampling
        source_states: Dict[str, Any] Dict from source to iterator state
        rng_state: dict numpy bit generator state used to resume rng
    """

    root_dir: str
    sources: Dict[str, float]
    source_to_state: Dict[str, Any]
    rng_state: Dict[str, Any]


class TokenizerState(TypedDict):
    it_state: Any
    name: str
    add_bos: bool
    add_eos: bool
    path: Optional[str]


@dataclass
class DataArgs:
    root_dir: Optional[str] = None
    sources: Dict[str, float] = field(default_factory=dict)
    batch_size: int = 2
    seq_len: int = 2048
    n_views: int = 2
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = True
    load_async: bool = True
    prefetch_size: int = 64
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)


class PackTokensState(TypedDict):
    """Represents the current state of a packing iterator.

    Attributes:
        start_token: int index to start reading from in the current sequence
        output_seq_len: int Length of sequences to output
        n_views: dict int Number of views to output. Each view is the same sequence but shifted by 1 from the previous
    """

    start_token: int
    it_state: Any
    output_seq_len: int
    n_views: int
    seq_len: int


def find_and_sanitize_chunks(dataset_path: str, world_size: int, file_pattern: str = "*.chunk.*.jsonl"):
    dataset_chunks = [str(p) for p in Path(dataset_path).glob(file_pattern)]
    n_chunks = len(dataset_chunks)

    if n_chunks > world_size:
        n_discard = n_chunks - world_size
        dataset_chunks = dataset_chunks[:world_size]
    else:
        assert (
                world_size % n_chunks == 0
        ), "World size should be a multiple of number of chunks"

    assert n_chunks > 0, f"No valid chunks in {dataset_path}"

    return dataset_chunks


def distribute_data_to_rank(dataset_path: str, rank: int, world_size: int, file_pattern: str):
    """
    Distributes the chunk files in a dataset path to each worker.
    If world_size is smaller than the number of chunks, the extra chunks are discarded.
    Otherwise, world_size is assumed to be a multiple of number of chunks.
    In that case there are world_size//nb_chunks workers on each chunk file, reading with different offsets.
    """
    dataset_chunks = find_and_sanitize_chunks(dataset_path, world_size, file_pattern)
    n_ranks_per_chunk = world_size // len(dataset_chunks)
    rank_to_jsonl_iterator_params = []
    for chunk_path in dataset_chunks:
        for i in range(n_ranks_per_chunk):
            rank_to_jsonl_iterator_params.append(
                JSONLState(
                    file_path=chunk_path,
                    position=0,
                    block_size=n_ranks_per_chunk,
                    offset=i,
                    current_iter=0,
                )
            )

    return rank_to_jsonl_iterator_params[rank]


def init_choice_state(
        root_dir: str,
        sources: Dict[str, float],
        seed: int,
        rank: int,
        world_size: int,
        file_pattern: str,
):
    data_path_to_jsonl_state = dict()
    for dataset_path in sources:
        jsonl_state = distribute_data_to_rank(
            os.path.join(root_dir, dataset_path), rank, world_size, file_pattern
        )
        data_path_to_jsonl_state[dataset_path] = jsonl_state

    multi_rng_state = np.random.default_rng(
        (seed, rank, world_size)
    ).bit_generator.state

    multi_choice_state = MultiChoiceState(
        root_dir=root_dir,
        sources=sources,
        source_to_state=data_path_to_jsonl_state,
        rng_state=multi_rng_state,
    )
    return multi_choice_state


def init_state(
        root_dir: str,
        sources: Dict[str, float],
        batch_size: int,
        prefetch_size: int,
        seq_len: int,
        n_views: int,
        seed: int,
        rank: int,
        world_size: int,
        add_bos: bool,
        add_eos: bool,
        tokenizer_name: str,
        tokenizer_path: Optional[str] = None,
        file_pattern: str = "*.chunk.*.jsonl"
):
    multi_choice_state = init_choice_state(
        root_dir=root_dir, sources=sources, seed=seed, rank=rank, world_size=world_size, file_pattern=file_pattern
    )
    tokenizer_state = TokenizerState(
        it_state=multi_choice_state,
        add_bos=add_bos,
        add_eos=add_eos,
        name=tokenizer_name,
        path=tokenizer_path,
    )
    pack_state = PackTokensState(
        start_token=0,
        it_state=tokenizer_state,
        output_seq_len=seq_len,
        n_views=n_views,
        seq_len=0,
    )
    prefetch_rng_state = np.random.default_rng(
        (seed + 1, rank, world_size)
    ).bit_generator.state

    return PrefetchState(
        it_state=pack_state,
        seq_idx=0,
        rng_state=prefetch_rng_state,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
    )


def init_dataloader_state_from_args(
        args: DataArgs,
        rank: int,
        world_size: int,
):
    return init_state(
        root_dir=args.root_dir,
        sources=args.sources,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        prefetch_size=args.prefetch_size,
        n_views=args.n_views,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        tokenizer_name=args.tokenizer.name,
        tokenizer_path=args.tokenizer.path,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )






class CustomDataset(Dataset):
    def __init__(self, token_ids, config: BaseConfiguration):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - config.model_max_sequence, config.strides):
            input_chunk = token_ids[i: i + config.model_max_sequence]
            target_chunk = token_ids[i + 1: i + config.model_max_sequence + 1]
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
