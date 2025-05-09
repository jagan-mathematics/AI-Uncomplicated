import torch
from torch.optim import AdamW
from typing import Optional, List

from core.trainer.schedulers import WarmupCosineScheduler


# todo: yet to be optimizer for multiple scheduler choice and optimizer choice
def get_optimzer_with_scheduler(
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
    """
    Create AdamW optimizer and warmup scheduler.

    Args:
        model: PyTorch model
        num_training_steps: Total number of training steps
        learning_rate: Maximum learning rate after warmup
        weight_decay: Weight decay coefficient
        warmup_ratio: Ratio of warmup steps to total steps
        min_lr_ratio: Minimum learning rate ratio compared to max lr
        beta1: AdamW beta1 parameter
        beta2: AdamW beta2 parameter
        eps: AdamW epsilon parameter
        correct_bias: Whether to correct bias in AdamW
        no_decay_params: List of parameter names that should not have weight decay

    Returns:
        tuple: (optimizer, scheduler)
    """
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

    warmup_ratio = None
    warmup_steps = -1
    if warmup_ratio > 0:
        warmup_steps = int(num_training_steps * warmup_ratio)
        print(f"warmup_steps :: {warmup_steps}")

        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio
        )

    return optimizer, scheduler, warmup_steps