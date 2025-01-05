import math
from typing import Callable, Dict, Optional
import torch
import torch.nn as nn


def calculate_gelu_gain():
    """
    Calculates the gain for GELU activation with tanh approximation.
    Uses numerical differentiation to find the variance preservation factor.
    """
    def gelu(x):
        # GELU with tanh approximation
        return x * 0.5 * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

    # Generate random input samples
    num_samples = 1000000
    x = torch.randn(num_samples)

    # Calculate numerical derivative
    epsilon = 1e-6
    x_plus = x + epsilon
    x_minus = x - epsilon
    derivative = (gelu(x_plus) - gelu(x_minus)) / (2 * epsilon)

    # Calculate variance of derivative (gain is square root of this)
    gain = float(torch.sqrt(torch.mean(derivative**2)))

    return gain


def get_initializer(
    init_type: str,
    activation: Optional[str] = None,
    embedding_init: Optional[str] = None,
    **kwargs
) -> Callable:
    """
    Returns a weight initialization function based on the specified type.

    Args:
        init_type: Type of initialization ('xavier', 'kaiming', 'normal', 'uniform', 'orthogonal')
        activation: Activation function used after the layer ('relu', 'leaky_relu', 'tanh', etc.)
        embedding_init: Special initialization for embedding layers ('normal', 'uniform', 'xavier', None)
        **kwargs: Additional arguments for the initialization function
            - embedding_mean: Mean for normal embedding initialization (default: 0.0)
            - embedding_std: Std for normal embedding initialization (default: 0.02)
            - embedding_padding_idx: Index for padding token

    Returns:
        Callable that initializes network parameters
    """


    def init_embedding_weights(embedding_layer: nn.Embedding, method: str) -> None:
        padding_idx = kwargs.get('embedding_padding_idx', None)

        if method == 'normal':
            mean = kwargs.get('embedding_mean', 0.0)
            std = kwargs.get('embedding_std', 0.02)
            nn.init.normal_(embedding_layer.weight, mean=mean, std=std)
        elif method == 'uniform':
            bound = kwargs.get('embedding_bound', 0.05)
            nn.init.uniform_(embedding_layer.weight, -bound, bound)
        elif method == 'xavier':
            nn.init.xavier_uniform_(embedding_layer.weight)

        # Zero out padding token embeddings if padding_idx is specified
        if padding_idx is not None:
            with torch.no_grad():
                embedding_layer.weight[padding_idx].fill_(0)


    def xavier_init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if activation in ['relu', 'leaky_relu']:
                gain = nn.init.calculate_gain('leaky_relu' if activation == 'leaky_relu' else 'relu')
            elif activation == "gelu":
                gain = 1.70093
            else:
                gain = nn.init.calculate_gain('tanh')

            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding) and embedding_init:
            init_embedding_weights(m, embedding_init)


    def kaiming_init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            mode = kwargs.get('mode', 'fan_in')
            nonlinearity = activation if activation else 'relu'

            nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding) and embedding_init:
            init_embedding_weights(m, embedding_init)

    def normal_init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            mean = kwargs.get('mean', 0.0)
            std = kwargs.get('std', 0.02)

            nn.init.normal_(m.weight, mean=mean, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding) and embedding_init:
            init_embedding_weights(m, embedding_init)

    def uniform_init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            a = kwargs.get('a', -0.05)
            b = kwargs.get('b', 0.05)

            nn.init.uniform_(m.weight, a=a, b=b)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding) and embedding_init:
            init_embedding_weights(m, embedding_init)

    def orthogonal_init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            gain = kwargs.get('gain', 1.0)

            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding) and embedding_init:
            init_embedding_weights(m, embedding_init)

    initializers: Dict[str, Callable] = {
        'xavier': xavier_init,
        'kaiming': kaiming_init,
        'normal': normal_init,
        'uniform': uniform_init,
        'orthogonal': orthogonal_init
    }

    if init_type not in initializers:
        raise ValueError(f"Initialization type '{init_type}' not supported. "
                        f"Choose from {list(initializers.keys())}")

    return initializers[init_type]
