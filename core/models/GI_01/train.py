"""
Available autowrap policy from torch 2.4.0
__all__ = [
    "always_wrap_policy",
    "lambda_auto_wrap_policy",
    "transformer_auto_wrap_policy",
    "size_based_auto_wrap_policy", # trying this now
    "enable_wrap",
    "wrap",
    "CustomPolicy",
    "ModuleWrapPolicy",
]
"""

import os
import sys
sys.path.append('/workspace/AI-Uncomplicated')
import torch
import math
from typing import List
from torch.optim import AdamW
import torch.distributed as dist
import torch.multiprocessing as mp


from core.models.GI_01.main.config import ModelConfig, DatasetConfig, TrainingConfig
from core.tokenizer import SPMTokenizer
from core.models.GI_01.main.model import ConstrueAutoRegressiveModel
from core.models.GI_01.main.data_loader import EntoPTDataSet, create_data_loader, NextTokenPredictionCollator

from core.trainer.logger.local_logger import TrainingLogger
from core.trainer.schedulers import WarmupCosineScheduler
from core.trainer.loss import cross_entropy_loss
from core.trainer.utils import get_initializer

from typing import Optional
import functools
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


# import torch.distributed as dist
# import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp.fully_sharded_data_parallel import (
#     ShardingStrategy,
#     MixedPrecision
# )
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def fsdp_train(
      rank: int = None,
      world_size: int = None,
      args = None,
    ):
    """
    Training via fsdp methodology
    """
    setup(rank, world_size)

    distributed_args = {
        "rank" : rank,
        "num_replicas" : world_size,
        "shuffle" : True
    }

    train_dataloader = create_data_loader(
        args.get("train_dataset"),
        collate_fn=args.get("collate_fn"),
        batch_size=args.get("batch_size"),
        is_distributed_env=True,
        distributed_sampler_args=distributed_args
    )

    num_training_steps = calculate_training_steps(
        len(train_dataloader),
        num_epochs=args.get("num_epochs"),
        batch_size=args.get("batch_size"),
        gradient_accumulation_steps=args.get("gradient_accumulation_steps",1)
    )

    print(f"==== PRETRAINING :: TOTAL TRAINING STEPS :: {num_training_steps}")

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=200
    )

    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = ConstrueAutoRegressiveModel(config=args.get("model_config")).to(rank)
    initalizer = get_initializer(init_type="xavier", activation="gelu", embedding_init="xavier", embedding_padding_idx=args.get("model_config").padding_id)

    model.apply(initalizer)
#     validate_model_initial_states(model, args.get("model_config"), (10, ), n_samples=1000)
#     check_model_value_range(model)

    model = FSDP(model)

    optimizer, scheduler, warmup_steps = create_optimizer_and_scheduler(
        model=model,
        num_training_steps=num_training_steps,
        learning_rate=args.get("learning_rate"),
        warmup_ratio=args.get("warmup_ratio"),
        weight_decay=args.get("weight_decay")
    )

    train_logger = TrainingLogger(
        project_name=args.get("experimentation_name"),
        output_dir=args.get("output_dir"),
        log_every_n_steps=args.get("logging_steps"),
        save_every_n_steps=args.get("save_steps"),
    )

    for epoch in range(1, args.get("num_epochs") + 1):
        desc = f'  - (Training)   Epoch {epoch}'
        n_word_total, n_word_correct, total_loss = 0, 0, 0
        for step, batch in tqdm(enumerate(train_dataloader), mininterval=2, desc=desc, leave=False, total=len(train_dataloader)):
            loss, n_correct, n_word = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                device="cuda",
                gradient_accumulation_steps=args.get("gradient_accumulation_steps",1),
            )
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss

            current_lr = scheduler.get_last_lr()[0]

            train_logger.log_metrics(metrics={
                'loss': loss,
                'epoch': epoch,
                "n_correct": n_correct,
                "n_word": n_word,
                'learning_rate': current_lr,
                "warmup_steps": warmup_steps
            }, step=step)
        print()
        print(f"Epoch {epoch} :: loss per word :: {total_loss/n_word_total}")




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



from tqdm import tqdm

def train_step(model, batch, optimizer, scheduler, device, logger=None, gradient_accumulation_steps=1):
    # forward
    labels = batch["labels"].to(device)
    optimizer.zero_grad()
    pred = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
    logits = pred["logits"]
    # backward and update parameters
    loss, n_correct, n_word = cal_performance(logits, labels)
    loss = loss / gradient_accumulation_steps

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item() * gradient_accumulation_steps, n_correct, n_word


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def train(experimentation_name,
          output_dir,
          logging_steps,
          save_steps,
          num_epochs,
          train_dataset,
          collate_fn,
          batch_size,
          gradient_accumulation_steps=1, # not yet implemented
          learning_rate=5e-5,
          warmup_ratio=0.2,
          weight_decay=0.0,
          device="cuda"):


    model.train()
    model.to(device)

    train_dataloader = create_data_loader(train_dataset, collate_fn=collate_fn, batch_size=batch_size)

    num_training_steps = calculate_training_steps(len(train_dataloader), num_epochs=num_epochs, batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps)

    # Create optimizer and scheduler
    optimizer, scheduler, warmup_steps = create_optimizer_and_scheduler(
        model=model,
        num_training_steps=num_training_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay
    )

    train_logger = TrainingLogger(
        project_name=experimentation_name,
        output_dir=output_dir,
        log_every_n_steps=logging_steps,
        save_every_n_steps=save_steps,
    )

    for epoch in range(num_epochs):
        desc = f'  - (Training)   Epoch {epoch}'
        n_word_total, n_word_correct, total_loss = 0, 0, 0
        for step, batch in tqdm(enumerate(train_dataloader), mininterval=2, desc=desc, leave=False, total=len(train_dataloader)):
            loss, n_correct, n_word = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )


            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss

            current_lr = scheduler.get_last_lr()[0]

            train_logger.log_metrics(metrics={
                'loss': loss,
                'epoch': epoch,
                "n_correct": n_correct,
                "n_word": n_word,
                'learning_rate': current_lr,
                "warmup_steps": warmup_steps
            }, step=step)


        print()
        print(f"Epoch {epoch} :: loss per word :: {total_loss/n_word_total}")
    train_logger.finish()
    init_end_event.record()

if __name__ == "__main__":


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

    dataset_config = DatasetConfig(dataset_path=".artifacts/dataset",
                                   dataset_shuffle=True, batch_size=2)
    training_config = TrainingConfig(tokenizer_path="/workspace/AI-Uncomplicated/core/models/GI_01/tokenzier/en_fr_combined_tokenizer/",
                                     warm_up=4000,
                                     logging_steps=747)

    ## Load model and tokenizer
    tokenizer = SPMTokenizer(training_config.tokenizer_path)

    model_config.vocabulary_size = tokenizer.vocab_size


    ## Loading Data set
    print("========="*20)
    print("Loading Dataset")
    print("========="*20)

    from datasets import Dataset, load_from_disk

    import kagglehub

    # Download latest version
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


    train_dataloader = create_data_loader(
        train_examples_pt,
        collate_fn=collate_fn,
        batch_size=dataset_config.batch_size,
        )

    print("========="*20)
    print("Data Loader Created Successfully")
    print("Sample Datapoint: ")
    print("========="*20)

    for batch in train_dataloader:
        print(batch)
        break


    for l, i in zip(batch["labels"][0].numpy().tolist()[:10], batch["input_ids"][0].numpy().tolist()[:10]):
        print(f"{i} -> {l}")


    args = {
        "experimentation_name": training_config.experimentation_name,
        "output_dir": training_config.save_path,
        "logging_steps": training_config.logging_steps,
        "save_steps": training_config.logging_steps,
        "num_epochs": 2,  # training_config.num_epochs,
        "train_dataset": train_examples_pt,
        "collate_fn": collate_fn,
        "batch_size": dataset_config.batch_size,
        "learning_rate": 3e-4,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps" : 1,
        "model_config" : model_config
    }

    WORLD_SIZE = torch.cuda.device_count()

    print(f"===== WORLD SIZE IS {WORLD_SIZE} ====")

    mp.spawn(
        fsdp_train,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True
    )