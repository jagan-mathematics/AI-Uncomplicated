from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.nn import functional as F

from core.configurations.base import BaseConfiguration
from core.layers.positional_embedding.rope_projector import RopePositionEmbedding
from core.models.GI_01.main.decoder import ConstrueDecoderLayer
from core.layers.norms import RMSNorm
from core.utils.masks import create_causal_mask, create_multi_type_causal_mask

from torch import nn


@dataclass
class GI01ModelArgs(BaseConfiguration):
    embedding_init: str = "xavier"
    init_type: str = "xavier"
    activation: str = "gelu"



def cross_entropy(pred, target, **kwargs):
    loss = F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )
    return loss


class ConstrueModel(nn.Module):
    def __init__(self, config: BaseConfiguration):
        super().__init__()

        self.config = config

        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
            padding_idx=config.padding_id
        )
        self.rope_position_projection = RopePositionEmbedding(
                hidden_dim=config.head_dim,
                max_positions=config.max_positions,
                base=config.rope_base
            )

        # Decoder layer stack
        self.decoder_layers = nn.ModuleList([
            ConstrueDecoderLayer(
                config
            )
            for _ in range(config.num_layers)
        ])

        # Layer Norm initialization
        self.final_layer_norm = RMSNorm(
            model_dimension=config.hidden_dim
        )

    def forward(self, input_tensor, attn_mask=None, token_idx=None, attn_impl="sdpa", output_attentions=False, output_hidden_states=False):

        bsz, seqlen = input_tensor.shape

        # if attn_mask is None:
        #     attn_mask = torch.ones_like(input_tensor)

        hidden_states = self.token_embeddings(input_tensor)


        positional_frequency = self.rope_position_projection(input_tensor=input_tensor, token_idx=token_idx)

        # causal_mask = create_causal_mask(
        #     attention_mask=attn_mask,
        #     shape=input_tensor.shape,
        #     dtype=hidden_states.dtype,
        #     device=hidden_states.device
        # )

        causal_mask = (
            attn_mask
            if attn_mask is not None
            else create_multi_type_causal_mask(seqlen, attn_impl)
        )

        output_attentions_weights = (hidden_states)
        layers_hidden_states = ()
        for decoder_layer in self.decoder_layers:
            hidden_states, attention_weight = decoder_layer(
                hidden_states,
                causal_mask,
                frequency_cis=positional_frequency,
                token_idx=token_idx,
                attn_impl=attn_impl,
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
            out_features=config.vocab_size,
            bias=False
        )

    def forward(self, input_ids, attention_mask=None, target=None, token_idx=None, attn_impl="sdpa", output_attentions=False, output_hidden_states=False):
        last_hidden_state, output_attention_weights, hidden_states = self.model(input_ids, attention_mask,
                                                                                token_idx=token_idx,
                                                                                attn_impl=attn_impl)
        logits = self.lm_head(last_hidden_state)
        output = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "attention_map": output_attention_weights if output_attentions else None,
            "hidden_states": hidden_states if output_hidden_states else None
        }
        if target is not None:
            output["loss"] = cross_entropy(logits, target)
        return output

def build_fsdp_grouping_plan(model_args: GI01ModelArgs) -> List[Tuple[str, bool]]:
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("model.token_embeddings", False))

    # Grouping by layers
    for i in range(model_args.num_layers):
        group_plan.append((f"model.decoder_layers.{i}", False))

    return group_plan