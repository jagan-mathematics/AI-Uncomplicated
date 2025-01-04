import torch
import torch.nn.init

from core.configurations.base import BaseConfiguration
from core.models.translator.decoder import ConstrueDecoderLayer
from core.layers.layer_norm import LayerNorm
from core.utils.masks import _update_causal_mask
from core.layers.positional_embedding.rope_projector import RopePositionEmbedding

from torch import nn


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


    def forward(self, input_tensor, attn_mask=None):
        if attn_mask is None:
            attn_mask = torch.ones_like(input_tensor)

        hidden_states = self.token_embeddings(input_tensor)

        causal_mask = _update_causal_mask(
            input_tensor=hidden_states,
            attention_mask=attn_mask
        )


        for decoder_layer in self.decoder_layers:
            hidden_states, _ = decoder_layer(
                hidden_states,
                causal_mask
            )

        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states



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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, input_tensor, attn_mask=None):
        hidden_states = self.model(input_tensor, attn_mask)
        logits = self.lm_head(hidden_states)

        if self.config.output_last_hidden_state:
            return {
                "logits": logits,
                "last_hidden_state": hidden_states
            }

        return {
            "logits": logits
        }

    def generate(self, input_ids, max_length=100, temperature=1.0):
        with torch.no_grad():
            for _ in range(max_length):
                output = self.forward(input_ids)

                # Next token logits
                next_token_logits = output[:, -1, :] / temperature

                # Sample next token
                next_token = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1),
                    num_samples=1
                )

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token.item() == self.config.padding_id:
                    break

        return input_ids