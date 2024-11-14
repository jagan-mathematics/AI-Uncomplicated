"""Decoder Module implementation"""

from torch import nn

from core.configurations.base import BaseConfiguration
from core.layers.attention import RopeAttention
from core.layers.layer_norm import LayerNorm
from core.layers.point_wise_projection import PointWiseGatedProjection
from core.utils.masks import _update_causal_mask


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, ffn_hidden, num_head, dropout):
        super(DecoderLayer, self).__init__()
        base_cfg = BaseConfiguration(
            hidden_dim=512,
            head_dim=64,
            num_heads=8,
            max_positions=512,
            intermediate_dim=2048
        )

        self.norm1 = LayerNorm(model_dimension=base_cfg.hidden_dim)
        self.self_attn = RopeAttention(
            config=base_cfg
        )
        self.dropout = nn.Dropout(p=dropout)

        self.norm2 = LayerNorm(model_dimension=base_cfg.hidden_dim)
        self.ffn = PointWiseGatedProjection(config=base_cfg)
        self.dropout2 = nn.Dropout(p=dropout)

    # def forward(self, dec, targ_mask): # POST LAYER NORMALIZATION
    #     _x = dec # input copy for norm
    #
    #     x , _ = self.self_attn(input_tensor=dec, attention_mask=targ_mask) #
    #     x = self.dropout(x)
    #     x = self.norm1(x + _x)
    #
    #     _x = x # input copy for norm
    #     x = self.ffn(x)
    #     x = self.dropout2(x)
    #     x = self.norm2(x + _x)
    #
    #     return x


    def forward(self, dec, targ_mask):
        x = self.norm1(dec) # Trying Pre Layer Norm as discussed
        # x = x.view(x.size(0), x.size(1), 8, 64)
        causal_mask = _update_causal_mask(dec, targ_mask)
        x, _ = self.self_attn(input_tensor=x, attention_mask=causal_mask)
        x = self.dropout(x)
        x = dec + x

        y = self.norm2(x)
        y = self.ffn(y)
        y = self.dropout(y)
        y = x + y

        return y





