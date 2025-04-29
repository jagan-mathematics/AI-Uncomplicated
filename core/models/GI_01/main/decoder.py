"""Decoder Module implementation"""

from torch import nn

from core.configurations.base import BaseConfiguration
from core.layers.attention import AttentionBlock, MultiTypeAttentionBlock
from core.layers.norms import RMSNorm
from core.layers.point_wise_projection import PointWiseGatedProjection


class ConstrueDecoderLayer(nn.Module):
    def __init__(self, base_cfg: BaseConfiguration):
        super().__init__()

        self.input_norm = RMSNorm(model_dimension=base_cfg.hidden_dim)
        self.self_attn = MultiTypeAttentionBlock(
            config=base_cfg
        )
        self.attention_dropout = nn.Dropout(p=base_cfg.attention_dropout)

        self.post_attention_norm = RMSNorm(model_dimension=base_cfg.hidden_dim)
        self.mlp = PointWiseGatedProjection(config=base_cfg)
        self.dropout2 = nn.Dropout(p=base_cfg.attention_dropout)


    def forward(self, hidden_state, attention_mask, frequency_cis, token_idx=None, attn_impl: str = "sdpa", output_attentions=False):
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
        hidden_state, self_attn_weights = self.self_attn(
            input_tensor=hidden_state,
            attention_mask=attention_mask,
            frequency_cis=frequency_cis,
            token_idx=token_idx,
            attn_impl=attn_impl,
            output_attentions=output_attentions,
        )
        hidden_state = self.attention_dropout(hidden_state)
        hidden_state = residual_x + hidden_state

        residual_x = hidden_state

        hidden_state = self.post_attention_norm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = self.dropout2(hidden_state)
        hidden_state = residual_x + hidden_state

        return hidden_state, self_attn_weights


    def init_weights(self, init_std=None, factor=1.0):
        self.self_attn.reset_parameters(init_std, factor)
        self.input_norm.reset_parameters()

        self.mlp.reset_parameters(init_std, factor)
        self.post_attention_norm.reset_parameters()