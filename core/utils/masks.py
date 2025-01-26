from typing import Optional, Tuple
import torch
from torch import Tensor


def _update_causal_mask(
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
):
    dtype, device = input_tensor.dtype, input_tensor.device
    input_rank = input_tensor.ndim

    if input_rank == 4:  # [B, H, S, S]
        batch_size, _, sequence_length, _ = input_tensor.shape
    elif input_rank == 3:  # [B, S, D]
        batch_size, sequence_length, _ = input_tensor.shape
    else:
        raise ValueError(f"Unsupported input tensor rank: {input_rank}. Expected 3 or 4.")

    min_dtype = torch.finfo(dtype).min
    target_sequence_length = attention_mask.shape[-1]

    causal_mask = torch.full((sequence_length, target_sequence_length), fill_value=1,
                             dtype=dtype,
                             device=device)

    if sequence_length > 1:
        causal_mask = torch.tril(causal_mask)

    if input_rank == 4:
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    else:
        causal_mask = causal_mask[None, :, :].expand(batch_size, -1, -1)

    padding_mask = attention_mask[:, None, None, :] if input_rank == 4 else attention_mask[:, None, :]

    if padding_mask.device != device:
        padding_mask = padding_mask.to(device=device)

    causal_mask = (causal_mask * padding_mask).to(dtype)
    causal_mask = 1.0 - causal_mask
    causal_mask = causal_mask.masked_fill(causal_mask == 1.0, min_dtype)
    causal_mask = causal_mask.contiguous()
    return causal_mask
