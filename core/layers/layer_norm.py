"""Implementing Layer Norm"""

import numbers
import torch
from torch import nn
import torch.nn.functional as F

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

