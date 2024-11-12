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
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RopePositionEmbedding(nn.Module):
    def __init__(self, hidden_dim, max_positions=2048, base=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base = base
        
        positions = torch.arange(max_positions, dtype=torch.float32)
        angular_freq = 1.0 / (
                    self.base ** (torch.arange(0, self.hidden_dim, 2, dtype=torch.int64).float() / self.hidden_dim)
                    )
        angular_freq = angular_freq.float().unsqueeze(0) # 1 x D/2
        positions = positions.unsqueeze(1) # S x 1
        rotatory_matrix = positions @ angular_freq # S x D/2
        
        self.register_buffer("rotatory_matrix", tensor=rotatory_matrix, persistent=False)
    
    @torch.no_grad()
    def forward(self, x):
        batch_size, sequence_length, dim = x.shape
        assert self.hidden_dim == dim
        device_type = x.device
        
        if self.rotatory_matrix.device != device_type:
            self.rotatory_matrix.device.to(device_type)
        
        freqs = self.rotatory_matrix[:sequence_length, :]
        freqs = freqs[None, :, :].expand(batch_size, -1, -1)
        
        with torch.autocast(device_type=device_type.type, enabled=False):
            emb = torch.cat((freqs, freqs), dim=-1)

            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        