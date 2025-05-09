from dataclasses import dataclass
from functools import partial
import math

import logging
from torch import nn
from torch.optim import AdamW, lr_scheduler

from core.trainer.schedulers import WarmUpLinearScheduler, WarmupCosineScheduler, WSDScheduler

logger = logging.getLogger()


@dataclass
class OptimArgs:
    lr: float = 3e-4
    weight_decay: float = 0.1
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95
    clip: float = 1.0

    scheduler: str = "cosine"
    warmup: int = 2000
    min_lr: float = 0.1
    cycle_length: float = 1.0
    cosine_theta: float = 1.0
    annealing_step: int = 1000
    decay_fraction: float = 0.1

    exp_factor: float = 0.5


def build_scheduler_fn(optimizer, args, n_steps):
    if args.scheduler == "Constant":
        lr_fn = lambda x: args.lr
    elif args.scheduler == "WrLinear":
        lr_fn = WarmUpLinearScheduler(
            optimizer=optimizer,
            warmup_step=args.warmup,
            total_steps=args.n_steps,
            min_lr=args.min_lr,
        )
    elif args.scheduler == "cosine":
        lr_fn = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=args.warmup,
            total_steps=n_steps,
            cycle_length=args.cycle_length,
            theta=args.cosine_theta,
            min_lr=args.min_lr,
        )
    elif args.scheduler == "wsd":
        lr_fn = WSDScheduler(
            optimizer=optimizer,
            warmup_step=args.warmup,
            total_steps=n_steps,
            cycle_length=args.cycle_length,
            decay_fraction=args.decay_fraction,
            max_lr=args.lr,
            min_lr=args.min_lr,
        )
    else:
        raise ValueError(f"un-recognized scheduler type `{args.scheduler}`")
    return lr_fn


def build_optimizer(model: nn.Module, args: OptimArgs, n_steps: int):
    logger.info("Starting build of optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.epsilon,
        fused=True,
    )
    # scheduler
    scheduler = build_scheduler_fn(optimizer, args, n_steps)
    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
