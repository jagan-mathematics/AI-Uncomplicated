import torch
from torch import nn
from core.configurations.base import BaseConfiguration

from core.activations.gelu import PytorchGELUTanh


class PointWiseProjection(nn.Module):
    """
    point wise project as native from `attention is all you need`
    with slight changes in activation relu -> gelu tanh approximation
    https://arxiv.org/pdf/1706.03762
    """
    def __init__(self, config: BaseConfiguration):
        super().__init__()
        self.up_projection = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_projection = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.act_func = PytorchGELUTanh()


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.down_projection(self.act_func(self.up_projection(input_tensor)))
    
    

class PointWiseGatedProjection(nn.Module):
    """
    point wise project as native from `attention is all you need`
    with slight changes in activation relu -> gelu tanh approximation
    https://arxiv.org/pdf/1706.03762
    """

    def __init__(self, config: BaseConfiguration):
        super().__init__()
        self.intermediate_size = config.intermediate_dim
        self.gate_projection = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_projection = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_projection = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        
        self.act_func = PytorchGELUTanh()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.down_projection(self.act_func(self.gate_projection(input_tensor)) * self.up_projection(input_tensor))

