"""Decoder Module implementation"""

from torch import nn

from core.configurations.base import BaseConfiguration
from core.layers.attention import RopeAttention
from core.layers.layer_norm import LayerNorm
from core.layers.point_wise_projection import PointWiseGatedProjection
from core.utils.masks import _update_causal_mask


class DecoderLayer(nn.Module):
    def __init__(self, base_cfg: BaseConfiguration):
        super(DecoderLayer, self).__init__()

        self.input_norm = LayerNorm(model_dimension=base_cfg.hidden_dim)
        self.self_attn = RopeAttention(
            config=base_cfg
        )
        self.attention_dropout = nn.Dropout(p=base_cfg.attention_dropout)

        self.post_attention_norm = LayerNorm(model_dimension=base_cfg.hidden_dim)
        self.mlp = PointWiseGatedProjection(config=base_cfg)
        self.dropout2 = nn.Dropout(p=base_cfg.attention_dropout)


    def forward(self, hidden_state, attention_mask, output_attentions=False):
        """
         https://arxiv.org/pdf/2002.04745 (PRE-Norm)


        x = embedding of each tokens (B x S x D)
        mask = self_attention autput (B x S x D)

        Self-attention sub-block 1
        x => norm(x) => n_x => att(n_x) => (a_x, a_score) => a_x + x => h_x

        MLP sub-block: 2

        h_x => norm(h_x) => n_hx => mlp(n_hx) => n_x => x_x + h_x => h_x
        """
        residual_x = hidden_state

        hidden_state = self.input_norm(hidden_state)

        hidden_states, self_attn_weights = self.self_attn(
            input_tensor=hidden_state,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.attention_dropout(hidden_states)
        hidden_states = residual_x + hidden_state
        # point forward inner bloc
        residual_x = hidden_state

        hidden_state = self.post_attention_norm(hidden_states)
        hidden_state = self.mlp(hidden_state)
        hidden_state = self.dropout2(hidden_state)
        hidden_state = residual_x + hidden_state
        return hidden_state, self_attn_weights

