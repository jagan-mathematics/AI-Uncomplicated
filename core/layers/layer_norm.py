"""Implementing Layer Norm"""

import torch
from torch import nn

class LayerNorm(nn.Module):
    """
    Class the contains LayerNorm
    This is same as transformers implementation
    """
    def __init__(self, model_dimension, epsilon=1e-12):
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(model_dimension))
        self.beta = nn.Parameter(torch.zeros(model_dimension))
        self.epsilon = epsilon


    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, unbiased=False, keepdim=True)

        # Layer normalization computation
        output = (x - mean) / torch.sqrt(variance + self.epsilon)
        output = self.gamma * output + self.beta
        return output

