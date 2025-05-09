from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from typing import Optional, Callable


class EntoPTDataSet(Dataset):
    def __init__(self, tensorflow_dataset):
        self.dataset = tensorflow_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]



def worker_init_fn(worker_id):
    torch.manual_seed(torch.initial_seed() + worker_id)

def create_data_loader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
    is_distributed_env=False,
    distributed_sampler_args=None
) -> DataLoader:
    """
    Create a PyTorch DataLoader with optimized settings.
    """
    if collate_fn is None:
        raise ValueError("collate_fn function not provided")

    if is_distributed_env:
        sampler = DistributedSampler(dataset, **distributed_sampler_args)
    elif shuffle:
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn
    )

    return loader



class NextTokenPredictionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.start = tokenizer.model.piece_to_id("<s>")
        self.end = tokenizer.model.piece_to_id("</s>")

        self.en_start = tokenizer.model.piece_to_id("<lang_en>")
        self.en_end = tokenizer.model.piece_to_id("</lang_en>")
        self.pt_start = tokenizer.model.piece_to_id("<lang_pt>")
        self.pt_end = tokenizer.model.piece_to_id("</lang_pt>")
        self.pad_token_idx = tokenizer.model.PieceToId("<pad>")

    def __call__(self, batch):
        input_ids = []
        labels = []

        for item in batch:
            french, english = item["fr"], item["en"]

            english_encoded = self.tokenizer.encode(
                english, return_type=None, add_special_tokens=False
            )["input_ids"][0]

            french_encoded = self.tokenizer.encode(
                french, return_type=None, add_special_tokens=False
            )["input_ids"][0]


            english_encoded = [self.start] + [self.en_start] +  english_encoded +  [self.en_end]
            french_encoded = [self.pt_start] + french_encoded + [self.pt_end] + [self.end]

            input_ids.append(
                english_encoded + french_encoded
            )
            labels.append(
                (english_encoded + french_encoded)[1::] + [self.pad_token_idx]
            )

        paddded_tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in input_ids], batch_first=True, padding_value=self.pad_token_idx).long()
        attention_mask = (paddded_tokens != self.pad_token_idx).to(torch.int32)

        target = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in labels], batch_first=True, padding_value=self.pad_token_idx).long()
        target = torch.where(attention_mask == 0, -100, target)

        return {"input_ids": paddded_tokens, "attention_mask": attention_mask, "labels": target}