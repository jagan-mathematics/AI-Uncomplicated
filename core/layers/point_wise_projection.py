import torch
from torch import nn
from core.configurations.base import BaseConfiguration

from core.activations.gelu import PytorchGELUTanh
import torch.nn.functional as F

class PointWiseProjection(nn.Module):
    """
    point wise project as native from `attention is all you need`
    with slight changes in activation relu -> gelu tanh approximation
    https://arxiv.org/pdf/1706.03762
    """
    def __init__(self, config: BaseConfiguration):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.intermediate_size = config.intermediate_dim

        self.up_projection = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_projection = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        # self.act_func = PytorchGELUTanh()


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.down_projection(F.silu(self.up_projection(input_tensor)))

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.hidden_dim ** (-0.5))
        out_init_std = init_std or (self.intermediate_size ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor

        nn.init.trunc_normal_(
            self.up_projection.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )

        nn.init.trunc_normal_(
            self.down_projection.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )



class PointWiseGatedProjection(nn.Module):
    """
    point wise project as native from `attention is all you need`
    with slight changes in activation relu -> gelu tanh approximation
    https://arxiv.org/pdf/1706.03762
    """

    def __init__(self, config: BaseConfiguration):
        super().__init__()
        self.hidden_dim = config.hidden_dim

        intermediate_dim = int(2 * config.intermediate_dim / 3)
        if config.ffn_dim_multiplier is not None:
            intermediate_dim = int(config.ffn_dim_multiplier * intermediate_dim)
        intermediate_dim = config.multiple_of * ((intermediate_dim + config.multiple_of - 1) // config.multiple_of)
        self.intermediate_dim = intermediate_dim

        # print(f"## config hidden dim {config.hidden_dim}")
        # print(f"## config intermediate dim {config.intermediate_dim}")
        self.gate_projection = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_projection = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_projection = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

        # self.act_func = PytorchGELUTanh()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.down_projection(F.silu(self.gate_projection(input_tensor)) * self.up_projection(input_tensor))


    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.hidden_dim ** (-0.5))
        out_init_std = init_std or (self.intermediate_dim ** (-0.5))
        in_init_std = in_init_std

        out_init_std = out_init_std / factor
        for w in [self.up_projection, self.gate_projection]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.down_projection.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )