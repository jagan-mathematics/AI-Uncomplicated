import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Optional, List

class WarmupCosineScheduler(LambdaLR):
    """Linear warmup and cosine decay scheduler."""

    def __init__(
        self,
        optimizer: AdamW,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1
    ):
        """
        Initialize warmup and decay scheduler.

        Args:
            optimizer: AdamW optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr_ratio: Minimum learning rate ratio compared to initial lr
            last_epoch: The index of the last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step: int) -> float:
        """Calculate lr multiplier based on current step."""
        if current_step < self.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, self.warmup_steps))

        # Cosine decay
        progress = float(current_step - self.warmup_steps) / \
            float(max(1, self.total_steps - self.warmup_steps))
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale decay to min_lr_ratio
        decay = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * decay
        return decay

