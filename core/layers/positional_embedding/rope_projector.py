import torch
from torch import nn


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

        rotatory_matrix = self._get_cache_rotatory_matrix(
            max_positions=self.max_positions,
            hidden_dim=self.hidden_dim,
            base=self.base
        )
        self.register_buffer("rotatory_matrix", tensor=rotatory_matrix, persistent=False)


    def _get_cache_rotatory_matrix(self, max_positions,
                             hidden_dim,
                             base: int = None):

        if base is None:
            base = 10000

        positions = torch.arange(max_positions, dtype=torch.float32)
        angular_freq = 1.0 / (
                base ** (torch.arange(0, hidden_dim, 2, dtype=torch.int64).float() / hidden_dim)
        )
        angular_freq = angular_freq.float().unsqueeze(0)  # 1 x D/2
        positions = positions.unsqueeze(1)  # S x 1
        rotatory_matrix = positions @ angular_freq  # S x D/2
        return rotatory_matrix



    @torch.no_grad()
    def forward(self, x):
        batch_size, _, sequence_length, dim = x.shape  # B x H x S x D
        assert self.hidden_dim == dim  # 64
        device_type = x.device

        if self.rotatory_matrix.device != device_type:
            self.rotatory_matrix.device.to(device_type)

        freqs = self.rotatory_matrix[:sequence_length, :]  # S x D / 2
        freqs = freqs[None, :, :].expand(batch_size, -1, -1)  # 1 x S x D/ 2 -> B x s x D/ 2

        with torch.autocast(device_type=device_type.type, enabled=False):
            emb = torch.cat((freqs, freqs), dim=-1)

            cos = emb.cos()  # B x S x D
            sin = emb.sin()  # B x S x D
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)  # B x S x D
