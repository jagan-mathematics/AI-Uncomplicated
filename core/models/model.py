"""Model layer class for LLM"""
import torch.nn.init
from sympy.physics.units import temperature

from core.configurations.base import BaseConfiguration
from core.layers.decoder.decoder import DecoderLayer
from core.layers.layer_norm import LayerNorm
from core.utils.masks import _update_causal_mask
from core.layers.positional_embedding.rope_projector import RopePositionEmbedding


from torch import nn


class LLM:
    def __init__(self, config: BaseConfiguration):
        super().__init__()

        self.config = config

        # token embedding layer
        self.token_embeddings = RopePositionEmbedding(
            hidden_dim=config.hidden_dim
        )

        # decoder layer stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                model_dim=config.hidden_dim,
                ffn_hidden=config.intermediate_dim,
                num_head=config.num_heads,
                dropout=config.attention_dropout
            )
            for _ in range(config.num_layers)
        ])

        # Layer Norm initialization
        self.final_layer_norm = LayerNorm(
            model_dimension=config.hidden_dim
        )

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, input_tensor, attn_mask=None):
        if attn_mask is None: # creating attn mask if it is empty
            attn_mask = torch.ones_like(input_tensor)

        causal_mask = _update_causal_mask(
            input_tensor=input_tensor,
            attention_mask=attn_mask
        )

        # getting hidden states
        hidden_states = self.token_embeddings(
            input_tensor
        )

        # Passing through decoder layer
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(
                hidden_states,
                causal_mask
            )

        # final Layer Norm NOTE: PreLayer Norm is also achieved and this is final layer Norm

        hidden_states = self.final_layer_norm(
            hidden_states
        )

        return hidden_states

    def generate(self, input_ids, max_length=100):

        with torch.no_grad():
            for _ in range(max_length):

                output = self.forward(
                    input_ids
                )

                # Next token logits
                next_token_logits = output[:, -1, :] / temperature

                # next token
                next_token = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1),
                    num_samples=1
                )

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token.item() == self.config.padding_id:
                    break

        return input_ids

