import functools
import math
import os
from typing import List, Optional
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP
)
import numbers
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from core.models.GI_01.main.config import DatasetConfig, ModelConfig, TrainingConfig
from core.models.GI_01.main.model import ConstrueAutoRegressiveModel
from core.models.GI_01.main.decoder import ConstrueDecoderLayer
from core.tokenizer.tokenizer_loader import SPMTokenizer
from core.models.GI_01.main.data_loader import EntoPTDataSet, NextTokenPredictionCollator

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

from datasets import Dataset, load_from_disk
import kagglehub

from core.trainer.logger.local_logger import TrainingLogger
from core.trainer.loss import cross_entropy_loss
from core.trainer.schedulers import WarmupCosineScheduler
from torch.optim import AdamW


def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = "localhost"
    # os.environ['MASTER_PORT'] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("complete init")

def cleanup():
    dist.destroy_process_group()


def worker_init_fn(worker_id):
    torch.manual_seed(torch.initial_seed() + worker_id)


def load_dataset(tokenizer, batch_size):
    # Download latest version
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    path = kagglehub.dataset_download("dhruvildave/en-fr-translation-dataset")

    print("Path to dataset files:", path)

    try:
        dataset = Dataset.from_csv(os.path.join(path, "en-fr.csv"))
        dataset = dataset.take(3000000)
        dataset = dataset.filter(lambda d: d["en"] is not None and d["fr"] is not None)
        dataset = dataset.filter(lambda x: len(tokenizer.encode(x["en"] + x["fr"], return_type=None)["input_ids"][0]) < 500)
        dataset = dataset.train_test_split(test_size=0.2)
        dataset.save_to_disk("filterd_dataset")
    except NameError:
        dataset = load_from_disk("filterd_dataset")


    train_examples_pt = EntoPTDataSet(dataset["train"])
    val_examples_pt = EntoPTDataSet(dataset["test"])

    collate_fn = NextTokenPredictionCollator(tokenizer=tokenizer)

    sampler_1 = DistributedSampler(train_examples_pt, rank=rank, num_replicas= world_size, shuffle=True)
    train_dataloader = DataLoader(
        dataset=train_examples_pt,
        batch_size=batch_size,
        sampler=sampler_1,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        shuffle=False
        # worker_init_fn=worker_init_fn
    )

    return train_dataloader, sampler_1


def cal_performance(pred, gold, trg_pad_idx=-100):
    ''' Apply label smoothing if needed '''

    loss = cross_entropy_loss(pred, gold, trg_pad_idx)

    pred = pred.max(-1)[1].view(-1)
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    num_training_steps: int,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    no_decay_params: Optional[List[str]] = None
):
    # Default params that should not have weight decay
    if no_decay_params is None:
        no_decay_params = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

    # Separate parameters that should and should not have weight decay
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_params)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_params)
            ],
            "weight_decay": 0.0,
        },
    ]

    # Create AdamW optimizer
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps
    )

    # Create scheduler with linear warmup and cosine decay
    print(f"NUM Training Step :: {num_training_steps}")

    warmup_steps = int(num_training_steps * warmup_ratio)
    print(f"warmup_steps :: {warmup_steps}")

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio
    )

    return optimizer, scheduler, warmup_steps


def calculate_training_steps(
    num_examples: int,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int = 1
) -> int:
    """
    Calculate the total number of training steps.

    Args:
        num_examples: Total number of training examples
        num_epochs: Number of epochs to train for
        batch_size: Batch size per forward pass
        gradient_accumulation_steps: Number of steps to accumulate gradients

    Returns:
        int: Total number of optimizer update steps
    """

    update_steps_per_epoch = math.ceil(num_examples / gradient_accumulation_steps)

    total_training_steps = update_steps_per_epoch * num_epochs

    return total_training_steps


import tqdm

def train_step(epoch, model, train_loader, optimizer, sampler, train_size):
    # forward
    model.train()
    rank = int(os.environ["RANK"])
    fsdp_loss = torch.zeros(2).to("cuda")
    if sampler:
        sampler.set_epoch(epoch)

    if rank==0:
        inner_pbar = tqdm.tqdm(
            total=train_size, colour="blue", desc="r0 Training Epoch"
        )

    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to("cuda")

        labels = batch["labels"]

        optimizer.zero_grad()
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = output["logits"]
        loss = cross_entropy_loss(logits, labels, -100)

        loss.backward()
        optimizer.step()

        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)

        if rank==0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]

    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy
    # return None


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print("rank ", rank)
    setup(rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    sample_input = torch.randint(0, 100, (4, 10))
    sample_input = sample_input.to(dtype=torch.long, device="cuda")


    model_config = ModelConfig(model_name="Construe", num_layers = 2,
                                padding_id = 0,
                                hidden_dim = 512,
                                intermediate_dim = 3072,
                                max_positions = 2048,
                                layer_norm_eps = 1e-05,
                                model_max_sequence = 2048,
                                num_heads = 8,
                                attention_dropout = 0.1)

    dataset_config = DatasetConfig(dataset_path=".artifacts/dataset", dataset_shuffle=True, batch_size=2)
    training_config = TrainingConfig(tokenizer_path="/workspace/AI-Uncomplicated/core/models/GI_01/tokenzier/en_fr_combined_tokenizer/", warm_up=4000, logging_steps=747)

    ## Load model and tokenizer
    tokenizer = SPMTokenizer(training_config.tokenizer_path)

    model_config.vocabulary_size = tokenizer.vocab_size

    model = ConstrueAutoRegressiveModel(model_config)
    model.to("cuda")


    train_dataloader, sampler_1 = load_dataset(tokenizer=tokenizer, batch_size=dataset_config.batch_size)

    training_size = len(train_dataloader)

    # for data in train_dataloader:
    #     break
    # print("============== Dataset stats ===================")
    # print(data["input_ids"].size())
    # print(data["attention_mask"].size())

    # print(model(input_ids=data["input_ids"].to('cuda'),
    #             attention_mask=data["attention_mask"].to('cuda'))["logits"].size())
    fsdp_model = FSDP(model, device_id=torch.cuda.current_device(),
        auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            ConstrueDecoderLayer,
        },
    ))
    print(fsdp_model)
    # fsdp_model.train()
    print(fsdp_model(sample_input)["logits"].size())


    num_training_steps = calculate_training_steps(
        len(train_dataloader),
        num_epochs=2,
        batch_size=dataset_config.batch_size,
        gradient_accumulation_steps=1
    )

    optimizer, scheduler, warmup_steps = create_optimizer_and_scheduler(
        model=fsdp_model,
        num_training_steps=num_training_steps,
        learning_rate=0.0001,
        warmup_ratio=0.1,
        weight_decay=0.0
    )

    train_logger = TrainingLogger(
        project_name="testing",
        output_dir="./",
        log_every_n_steps=100,
        save_every_n_steps=100,
    )

    print("all set to run")
    for epoch in range(2):
        print(f"Running Epoch :: {epoch}")
        train_step(epoch=epoch, model=fsdp_model, train_loader=train_dataloader, optimizer=optimizer, sampler=sampler_1, train_size=training_size)
        scheduler.step()

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()