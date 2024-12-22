"""
Data Loader Module
"""
import torch

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


