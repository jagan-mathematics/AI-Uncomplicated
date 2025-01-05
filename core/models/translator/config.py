from dataclasses import dataclass
import os
from core.configurations.base import BaseConfiguration
import randomname

@dataclass
class TrainingConfig:
    tokenizer_path: str
    experimentation_name: str = randomname.get_name()
    save_path: str = None
    weight_decay: float = 0.001
    warm_up: int = 0
    learning_rate: float = 2e-5
    num_epochs: int = 4
    eval_frequency: int = 1
    eval_iter: int = 1
    optimzer = "adam"

    def __post_init__(self):
        self.save_path = os.path.join(os.getcwd(), self.experimentation_name)
        if self.save_path is None:
            os.mkdir(self.save_path)


@dataclass
class OptimzerConfig:
    beta1 = 0.99
    beta2 = 0.98


@dataclass
class DatasetConfig:
    dataset_path: str
    batch_size: int = 32
    dataset_shuffle: bool = False


@dataclass
class ModelConfig(BaseConfiguration):
    pass
