from core.configurations.base import BaseConfiguration
from core.layers.positional_embedding.rope_projector import RopePositionEmbedding
from core.layers.positional_embedding.rope_projector import apply_positional_embedding
from torch import nn
import torch

import math

from core.utils.masks import _update_causal_mask


class RopeAttention(nn.Module):
    def __init__(self, config: BaseConfiguration):
        super(RopeAttention, self).__init__()
        assert config.head_dim is not None
        if config.hidden_dim % config.head_dim != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.attention_dropout = config.attention_dropout
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.scaling = 1 / math.sqrt(config.head_dim)
        
        self.use_rope = config.use_rope
        
        if config.use_rope:
            self.rope_position_projection = RopePositionEmbedding(
                hidden_dim=self.head_dim,
                max_positions=config.max_positions,
                base=config.rope_base
            )
        
        self.query_projecton = nn.Linear(in_features=config.hidden_dim, 
                                    out_features=config.hidden_dim,
                                    bias=False)
        self.key_projecton = nn.Linear(in_features=config.hidden_dim, 
                                    out_features=config.hidden_dim,
                                    bias=False)
        self.value_projecton = nn.Linear(in_features=config.hidden_dim, 
                                    out_features=config.hidden_dim,
                                    bias=False)
        
        self.output_projecton = nn.Linear(in_features=config.hidden_dim, 
                                    out_features=config.hidden_dim,
                                    bias=False)

        

    def forward(self, input_tensor, attention_mask, output_attentions=False):
        print(f"input tensor shape in attention forward is {input_tensor.shape}")
        b_size, seq_len, _ = input_tensor.shape # [4, 10, 512] [B * S * D]
        
        query_state = self.query_projecton(input_tensor) # [B * S * D]
        key_state = self.key_projecton(input_tensor) # [B * S * D]
        value_state = self.value_projecton(input_tensor) # [B * S * D]

        query_state = query_state.view(b_size, seq_len, self.num_heads,  self.head_dim).transpose(1, 2) # [B * H * S * D]
        key_state = key_state.view(b_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [B * H * S * D]
        value_state = value_state.view(b_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [B * H * S * D]
        
        if self.use_rope:
            cos, sin = self.rope_position_projection(query_state) # B x S x D
            query_state, value_state = apply_positional_embedding(query_state, value_state, cos, sin)
        
        attn_weights = torch.matmul(query_state, key_state.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, :key_state.shape[-2]]  # B x H x Q_s x K_s
            attn_weights = attn_weights * attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_state.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_state)
        
        if attn_output.size() != (b_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(b_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
            
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(b_size, seq_len, -1)
        attn_output = self.output_projecton(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
