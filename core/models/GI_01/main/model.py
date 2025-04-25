from dataclasses import dataclass

import torch
import torch.nn.init

from core.configurations.base import BaseConfiguration
from core.models.GI_01.main.decoder import ConstrueDecoderLayer
from core.layers.layer_norm import LayerNorm
from core.utils.masks import create_causal_mask

from torch import nn


@dataclass
class GI01ModelArgs(BaseConfiguration):
    pass


class ConstrueModel(nn.Module):
    def __init__(self, config: BaseConfiguration):
        super().__init__()

        self.config = config

        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocabulary_size,
            embedding_dim=config.hidden_dim,
            padding_idx=config.padding_id
        )

        # Decoder layer stack
        self.decoder_layers = nn.ModuleList([
            ConstrueDecoderLayer(
                config
            )
            for _ in range(config.num_layers)
        ])

        # Layer Norm initialization
        self.final_layer_norm = LayerNorm(
            model_dimension=config.hidden_dim
        )

    def forward(self, input_tensor, attn_mask=None, output_attentions=False, output_hidden_states=False):
        if attn_mask is None:
            attn_mask = torch.ones_like(input_tensor)

        hidden_states = self.token_embeddings(input_tensor)
        causal_mask = create_causal_mask(
            attention_mask=attn_mask,
            shape=input_tensor.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        output_attentions_weights = (hidden_states)
        layers_hidden_states = ()
        for decoder_layer in self.decoder_layers:
            hidden_states, attention_weight = decoder_layer(
                hidden_states,
                causal_mask,
                output_attentions=output_attentions
            )
            if output_hidden_states:
                layers_hidden_states += (hidden_states,)
            if output_attentions:
                output_attentions_weights += (attention_weight,)

        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, (output_attentions_weights if output_attentions else None), (
            layers_hidden_states if output_hidden_states else None)


class ConstrueAutoRegressiveModel(nn.Module):
    def __init__(self, config: BaseConfiguration):
        super().__init__()
        self.config = config
        self.model = ConstrueModel(config)
        self.lm_head = nn.Linear(
            in_features=config.hidden_dim,
            out_features=config.vocabulary_size,
            bias=False
        )

    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False):
        last_hidden_state, output_attention_weights, hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(last_hidden_state)

        return {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "attention_map": output_attention_weights if output_attentions else None,
            "hidden_states": hidden_states if output_hidden_states else None
        }
