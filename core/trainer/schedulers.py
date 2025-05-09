from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math

class WarmupCosineScheduler(LambdaLR):
    """Linear warmup and cosine decay scheduler."""

    def __init__(
        self,
        optimizer: AdamW,
        warmup_steps: int,
        cycle_length: int,
        theta: int,
        total_steps: int,
        min_lr: float = 0.1,
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
        self.cycle_length = cycle_length
        self.theta = theta
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step: int) -> float:
        """Calculate lr multiplier based on current step."""
        sign = ((current_step // (self.total_steps * self.cycle_length)) % 2) * -2 + 1
        if current_step < self.warmup_steps:
            return float(current_step) / self.warmup_steps
        elif current_step <= self.total_steps:
            s = float(current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (1 - self.min_lr) * (sign * math.cos(math.pi * s ** self.theta / self.cycle_length) + 1)

        else:
            return self.min_lr



class WarmUpLinearScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, warmup_step: int, total_steps: int, min_lr: int = 0.1, last_epoch: int = -1):
        self.warmup_step = warmup_step
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step: int) -> float:
        if current_step < self.warmup_step:
            return float(current_step) / self.warmup_step
        elif self.warmup_step <= current_step <= self.total_steps:
            value = float(current_step - self.warmup_step) / (self.total_steps - self.warmup_step)
            return value * self.min_lr + (1 - value)
        else:
            return self.min_lr



class WSDScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer,
                 warmup_step: int,
                 total_steps: int,
                 cycle_length: int,
                 decay_fraction: float,
                 max_lr: float,
                 min_lr: int = 0.1,
                 last_epoch: int = -1):
        self.warmup_step = warmup_step
        self.total_steps = total_steps
        self.cycle_length = cycle_length
        self.decay_fraction = decay_fraction
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.last_epoch = last_epoch
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step: int) -> float:
        cycle_num = current_step // int(self.total_steps * self.cycle_length) + 1
        curr_n_steps = int(self.total_steps * self.cycle_length) * cycle_num
        decay_length = int(curr_n_steps * self.decay_fraction)
        if current_step == self.total_steps:
            cycle_num -= 1
            curr_n_steps = self.total_steps

        if current_step < self.warmup_step:
            lr = self.max_lr * (float(current_step) / self.warmup_step)
        elif current_step <= curr_n_steps - decay_length:
            lr = self.max_lr
        elif curr_n_steps - decay_length < current_step <= curr_n_steps:
            step_in_decay = current_step - (curr_n_steps - decay_length)
            progress = step_in_decay / decay_length
            lr = 1 / ((progress * (1 / self.min_lr)) + ((1 - progress) * (1 / self.max_lr)))
        else:
            lr = self.min_lr
        return lr


