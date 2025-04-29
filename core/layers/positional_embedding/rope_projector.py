import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    assert x.shape[-1] % 2 == 0

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_positional_embedding(q, k, cos, sin):
    if len(cos.shape) == 3 and len(q.shape) == 4:
        cos = cos.unsqueeze(1)  # B x 1 x S x D
        sin = sin.unsqueeze(1)  # B X 1 X S X D
    q = (q * cos) + (rotate_half(q) * sin)  # (B x H x S x D ) * B x 1 x S  D
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RopePositionEmbedding(nn.Module):
    def __init__(self, hidden_dim, max_positions, base):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_positions = max_positions
        self.base = base

        frequency_cis = self._get_cache_rotatory_matrix(
            max_positions=self.max_positions,
            hidden_dim=self.hidden_dim,
            base=self.base
        )
        self.register_buffer("frequency_cis", tensor=frequency_cis, persistent=False)


    def _get_cache_rotatory_matrix(self, max_positions,
                             hidden_dim,
                             base: int = None):

        if base is None:
            base = 10000

        positions = torch.arange(max_positions, dtype=torch.float32)
        angular_freq = 1.0 / (
                base ** (torch.arange(0, hidden_dim, 2, dtype=torch.int64).float() / hidden_dim)
        )
        angular_freq = angular_freq.float() # .unsqueeze(0)  # 1 x D/2
        # positions = positions.unsqueeze(1)  # S x 1
        rotatory_matrix = torch.outer(positions, angular_freq).float() # S x D/2
        cos, sin = rotatory_matrix.cos(), rotatory_matrix.sin()
        return torch.stack((cos, -sin, sin, cos), dim=-1).view(*rotatory_matrix.size(), 2, 2) # S x D/2 x 2 x 2 for each dimension we have 2x2 matrix which is [[cos -sin], [sin cos]]



    @torch.no_grad()
    def forward(self, input_tensor, token_idx=None):
        assert input_tensor.ndim in [2, 3], "Input tensor should be of shape (B x S) or (B x S x D)"
        sequence_length = input_tensor.shape[1] if token_idx is None else None

        assert (sequence_length is not None) or (token_idx is not None), "Should provide atleast sequence_length or token_idx"

        device_type = input_tensor.device

        if self.frequency_cis.device != device_type:
            logger.info("Changing frequency_cis device to match input tensor")
            self.frequency_cis = self.frequency_cis.to(device_type)
        else:
            logger.info("Loaded properly")

        if token_idx is not None:
            return self.frequency_cis[token_idx]
        elif sequence_length is not None:
            return self.frequency_cis[0:sequence_length]


    def init_weights(self):
        frequency_cis = self._get_cache_rotatory_matrix(
            max_positions=self.max_positions,
            hidden_dim=self.hidden_dim,
            base=self.base
        )
        self.register_buffer("frequency_cis", tensor=frequency_cis, persistent=False)
