from typing import Optional, Tuple
import torch
from torch import Tensor


def create_causal_mask(
    attention_mask,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device
) -> Tensor:
    """
    Efficiently creates a causal mask without redundant computations.
    """
    seq_len = shape[-1]
    batch_size = shape[0]
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=dtype, device=device),
        diagonal=1
    )

    # Directly create with -inf instead of using masked_fill
    causal_mask = mask.masked_fill(mask > 0, float('-inf'))

    # Expand to match input shape
    if len(shape) == 4:  # [B, H, S, S]
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
    else:  # [B, S, S]
        causal_mask = causal_mask.view(1, seq_len, seq_len)

    if attention_mask is not None:
        if len(shape) == 4:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
        else:
            attention_mask = attention_mask.view(batch_size, 1, seq_len)

        mask_value = torch.finfo(dtype).min
        causal_mask = causal_mask.masked_fill(attention_mask == 0, mask_value)
    return causal_mask