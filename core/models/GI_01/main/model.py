from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F

from core.configurations.base import BaseConfiguration
from core.layers.positional_embedding.rope_projector import RopePositionEmbedding
from core.models.GI_01.main.decoder import ConstrueDecoderLayer
from core.layers.norms import RMSNorm
from core.utils.masks import create_causal_mask, create_multi_type_causal_mask

from torch import nn

class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


class TiedLinear(nn.Module):
    def __init__(self, tied_module: nn.Module) -> None:
        super().__init__()
        self.tied_module = tied_module
        if not hasattr(tied_module, "weight"):
            raise AttributeError(
                "Provided module does not have attribute 'weight'. Please check your tied_module."
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.tied_module.weight)

@dataclass
class GI01ModelArgs(BaseConfiguration):
    weight_tying: bool = False
    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256
    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"


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
        self.hidden_dim = config.hidden_dim
        self.init_base_std = config.init_base_std
        self.init_std_factor = InitStdFactor(config.init_std_factor)

        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
            padding_idx=config.padding_id
        )
        self.rope_embeddings = RopePositionEmbedding(
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

        self.weight_tying = config.weight_tying
        if config.weight_tying:
            self.llm_head = TiedLinear(self.token_embeddings)
        else:
            self.llm_head = nn.Linear(
                config.hidden_dim,
                config.vocab_size,
                bias=False,
            )


    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embeddings.reset_parameters()


    def init_weights(self, init_std=None):
        self.reset_parameters()

        init_std = init_std or (self.hidden_dim ** (-0.5))
        self.final_layer_norm.reset_parameters()

        nn.init.trunc_normal_(
            self.token_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.llm_head.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


        for depth, layer in enumerate(self.decoder_layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.decoder_layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.hidden_dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)




    def forward(self, input_ids, attention_mask=None, target=None, token_idx=None, attn_impl="sdpa", output_attentions=False, output_hidden_states=False):

        bsz, seqlen = input_ids.shape

        # if attn_mask is None:
        #     attn_mask = torch.ones_like(input_tensor)

        hidden_states = self.token_embeddings(input_ids)


        positional_frequency = self.rope_embeddings(seqlen=seqlen, token_idx=token_idx)

        # causal_mask = create_causal_mask(
        #     attention_mask=attn_mask,
        #     shape=input_tensor.shape,
        #     dtype=hidden_states.dtype,
        #     device=hidden_states.device
        # )

        causal_mask = (
            attention_mask
            if attention_mask is not None
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

        logits = self.llm_head(hidden_states)
        output = {
            "logits": logits,
            "last_hidden_state": hidden_states,
            "attention_map": output_attentions_weights if output_attentions else None,
            "hidden_states": hidden_states if output_hidden_states else None
        }
        if target is not None:
            output["loss"] = cross_entropy(logits, target)
        return output



def build_fsdp_grouping_plan(model_args: GI01ModelArgs) -> List[Tuple[str, bool]]:
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("token_embeddings", False))

    # Grouping by layers
    for i in range(model_args.num_layers):
        group_plan.append((f"decoder_layers.{i}", False))

    group_plan.append((f"llm_head", False))
    return group_plan