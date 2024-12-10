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
        """
        Goal of Layer norm is to have 0 mean and variance as 1
        """
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, unbiased=False, keepdim=True) # '-1' means last dimension.(model dimension)

        # computing layer norm across model dimension for each and every sample in batch

        output = (x - mean) / torch.sqrt(variance + self.epsilon)
        output = self.gamma * output + self.beta
        return output
