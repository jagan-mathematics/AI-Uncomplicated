from dataclasses import dataclass

@dataclass
class BaseConfiguration(object):
    model_name: str = None
    num_layers: int = None
    padding_id: int = None
    hidden_dim: int = None
    intermediate_dim: int = None
    max_positions: int = None
    vocabulary_size: int = None
    layer_norm_eps: float = 1e-05
    model_max_sequence: int = 2048
    num_heads: int = 8
    attention_dropout = 0.0
    head_dim = None


    def __post_init__(self):
        if self.head_dim is None:
            assert self.hidden_dim % self.num_heads == 0
            self.head_dim = self.hidden_dim // self.num_heads


    def get_padding_token(self):
        """get model padding tokens"""
        return self.padding_id