from dataclasses import dataclass

@dataclass
class BaseConfiguration(object):
    """
    base config for model architecture
    """
    name: str = None
    num_layers: int = 1
    padding_id: int = 0
    hidden_dim: int = 512
    intermediate_dim: int = 3072
    max_positions: int = 2048
    vocabulary_size: int = -1
    layer_norm_eps: float = 1e-05
    max_seq_len: int = 2048
    num_heads: int = 8
    attention_dropout: int = 0.0
    head_dim: int = None
    use_rope: bool = True
    rope_base: float = 10000.0
    output_last_hidden_state: bool = False
    seed: int = 42

    def __post_init__(self):
        if self.head_dim is None:
            assert self.hidden_dim % self.num_heads == 0
            self.head_dim = self.hidden_dim // self.num_heads

        if self.vocabulary_size is None:
            raise ValueError("Vocabulary size should not be empty")

    def get_padding_token(self):
        """get model padding tokens"""
        return self.padding_id
