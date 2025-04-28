"""Implementing Layer Norm"""

import numbers
import torch
from torch import nn
import torch.nn.functional as F
from core.trainer import probe


class LayerNorm(nn.Module):
    """
    Class the contains LayerNorm
    This is same as transformers implementation
    """
    def __init__(self, model_dimension, epsilon=0.00001):
        super(LayerNorm, self).__init__()
        if isinstance(model_dimension, numbers.Integral):
            # mypy error: incompatible types in assignment
            model_dimension = (model_dimension,)
        self.model_dimension = model_dimension
        self.gamma = nn.Parameter(torch.ones(model_dimension))
        self.beta = nn.Parameter(torch.zeros(model_dimension))
        self.epsilon = epsilon


    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        return F.layer_norm(
            x, self.model_dimension, self.gamma, self.beta, self.epsilon
        )



class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, model_dimension: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(model_dimension))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        x = probe.log_stats(x, "resid")
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def init_weights(self):
        torch.nn.init.ones_(self.weight)  # type: ignore