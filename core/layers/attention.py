from typing import Optional, Tuple
from core.configurations.base import BaseConfiguration
from core.layers.positional_embedding.rope_projector import RopePositionEmbedding
from core.layers.positional_embedding.rope_projector import apply_positional_embedding
from torch import nn
import torch.nn.functional as F
import torch

import math

from xformers.ops import fmha, AttentionBias


from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention
)

flex_attention_comp = torch.compile(flex_attention)



def reshape_for_broadcast(frequency_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        frequency_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert frequency_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"frequency_cis vs x: {(frequency_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return frequency_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    frequency_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    frequency_cis = reshape_for_broadcast(
        frequency_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * frequency_cis).sum(5).flatten(3)
    xk_out = (xk_ * frequency_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class AttentionBlock(nn.Module):
    """
    Implements attention mechanism with Rotary Position Embedding (RoPE).
    """

    def __init__(self, config: BaseConfiguration):
        super().__init__()
        assert config.head_dim is not None
        if config.hidden_dim % config.head_dim != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scaling = 1 / math.sqrt(config.head_dim)

        # Initialize projection layers
        self.qkv_projection = nn.Linear(
            config.hidden_dim,
            3 * config.hidden_dim,
            bias=False
        )

        self.output_projection = nn.Linear(
            config.hidden_dim,
            config.hidden_dim,
            bias=False
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        frequency_cis: Optional[torch.Tensor] = None,
        token_idx: int = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for RoPE attention.

        Args:
            input_tensor: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            positions: Optional position indices
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        if input_tensor.dim() != 3:
            raise ValueError(f"Expected 3D input, got {input_tensor.dim()}D")

        batch_size, seq_length, _ = input_tensor.shape

        # Fused QKV projection
        qkv = self.qkv_projection(input_tensor)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        query_states, key_states, value_states = qkv
        # Apply RoPE if enabled
        query_states, key_states = apply_rotary_emb(query_states, key_states, 2 ,frequency_cis[0:seq_length])

        if hasattr(self, "kv_cache"):
            key_states, value_states = self.kv_cache.update(key_states, value_states, token_idx)

        # Compute attention scores with improved numerical stability
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_weights = attn_weights * self.scaling

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask to proper dtype
            attention_mask = attention_mask.to(dtype=query_states.dtype)

            # Add head dimension if needed
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

            # Ensure proper sequence length
            attention_mask = attention_mask[:, :, :, :key_states.shape[-2]]

            attn_weights = attn_weights + attention_mask

        # Compute attention probabilities with improved numerical stability
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.to(query_states.dtype)

        # Apply dropout during training
        if self.training:
            attn_weights = F.dropout(
                attn_weights,
                p=self.attention_dropout,
                training=True
            )

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)

        # Validate output shape
        expected_shape = (batch_size, self.num_heads, seq_length, self.head_dim)
        if attn_output.shape != expected_shape:
            raise ValueError(
                f"Expected output shape {expected_shape}, got {attn_output.shape}"
            )

        # Reshape output and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_dim)
        attn_output = self.output_projection(attn_output)

        return attn_output, (attn_weights if output_attentions else None)




class MultiTypeAttentionBlock(nn.Module):
    """
    Implements attention mechanism with Rotary Position Embedding (RoPE).
    """

    def __init__(self, config: BaseConfiguration):
        super().__init__()
        assert config.head_dim is not None
        if config.hidden_dim % config.head_dim != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scaling = 1 / math.sqrt(config.head_dim)

        # Initialize projection layers
        self.qkv_projection = nn.Linear(
            config.hidden_dim,
            3 * config.hidden_dim,
            bias=False
        )

        self.output_projection = nn.Linear(
            config.hidden_dim,
            config.hidden_dim,
            bias=False
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        frequency_cis: Optional[torch.Tensor] = None,
        token_idx: int = None,
        attn_impl: str = "sdpa",
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for RoPE attention.

        Args:
            input_tensor: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            positions: Optional position indices
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        if input_tensor.dim() != 3:
            raise ValueError(f"Expected 3D input, got {input_tensor.dim()}D")

        batch_size, seq_length, _ = input_tensor.shape
        output_shape = input_tensor.shape
        # Fused QKV projection
        qkv = self.qkv_projection(input_tensor)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # [3, batch_size, seq_len, num_heads,  head_dim]


        query_states, key_states, value_states = qkv
        # Apply RoPE if enabled
        query_states, key_states = apply_rotary_emb(query_states, key_states, 1 ,frequency_cis)

        if hasattr(self, "kv_cache"):
            key_states, value_states = self.kv_cache.update(key_states, value_states, token_idx)

        if attn_impl == "flex_attention":
            assert attention_mask is None or isinstance(attention_mask, BlockMask)
            query_states, key_states, value_states = map(lambda e: e.transpose(1, 2), (query_states, key_states, value_states))
            output = flex_attention_comp(query_states, key_states, value_states, block_mask=attention_mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "fmha":
            assert attention_mask is None or isinstance(attention_mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=attention_mask)
            # This uses B S H D instead of B H S D of pytorch

        elif attn_impl == "sdpa":
            query_states, key_states, value_states = map(lambda e: e.transpose(1, 2), (query_states, key_states, value_states))
            assert attention_mask is None or isinstance(attention_mask, (str, torch.Tensor))
            is_causal = (attention_mask == "causal") if isinstance(attention_mask, str) else False
            attention_mask = attention_mask if isinstance(attention_mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=is_causal,
                attn_mask=attention_mask,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        attn_output = self.output_projection(output.reshape(output_shape))
        return attn_output, None


