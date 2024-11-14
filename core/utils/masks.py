import torch

def _update_causal_mask(
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
):
    dtype, device = input_tensor.dtype, input_tensor.device
    batch_size, sequence_length, _ = input_tensor.shape
    
    min_dtype = torch.finfo(dtype).min
    
    target_sequence_length = attention_mask.shape[-1]

    casual_mask = torch.full((sequence_length, target_sequence_length), fill_value=1,
                         dtype=dtype,
                         device=device)
    
    if sequence_length > 1:
        casual_mask = torch.tril(casual_mask)

    casual_mask = casual_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    padding_mask = attention_mask[:, None, None, :]
    
    if padding_mask.device != device:
        padding_mask = padding_mask.to(device=device)
    
    casual_mask = (casual_mask * padding_mask).to(dtype)
    casual_mask = 1.0 - casual_mask
    casual_mask = casual_mask.masked_fill(casual_mask == 1.0, min_dtype)
    casual_mask = casual_mask.contiguous()
    return casual_mask