from typing import Optional, Tuple
from core.configurations.base import BaseConfiguration
from core.layers.positional_embedding.rope_projector import RopePositionEmbedding
from core.layers.positional_embedding.rope_projector import apply_positional_embedding
from torch import nn
import torch.nn.functional as F
import torch

import math


class RopeAttention(nn.Module):
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

        # Initialize RoPE if enabled
        self.use_rope = config.use_rope
        if self.use_rope:
            self.rope_position_projection = RopePositionEmbedding(
                hidden_dim=config.head_dim,
                max_positions=config.max_positions,
                base=config.rope_base
            )
        
        self.query_projection = nn.Linear(in_features=config.hidden_dim, 
                                    out_features=config.hidden_dim,
                                    bias=False)
        self.key_projection = nn.Linear(in_features=config.hidden_dim, 
                                    out_features=config.hidden_dim,
                                    bias=False)
        self.value_projection = nn.Linear(in_features=config.hidden_dim, 
                                    out_features=config.hidden_dim,
                                    bias=False)
        
        self.output_projection = nn.Linear(in_features=config.hidden_dim, 
                                    out_features=config.hidden_dim,
                                    bias=False)

        # Initialize projection layers
        self.qkv_projection = nn.Linear(
            config.hidden_dim,
            3 * config.hidden_dim,
            bias=False
        )

    def forward(self, input_tensor, attention_mask, output_attentions=False):
        b_size, seq_len, _ = input_tensor.shape
        
        query_state = self.query_projecton(input_tensor) # [B * S * D]
        key_state = self.key_projecton(input_tensor) # [B * S * D]
        value_state = self.value_projecton(input_tensor) # [B * S * D]

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
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
        if self.use_rope:
            cos, sin = self.rope_position_projection(query_states)
            query_states, key_states = apply_positional_embedding(
                query_states, key_states, cos, sin
            )

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
