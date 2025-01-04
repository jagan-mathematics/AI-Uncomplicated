from typing import Callable, Dict, Optional
import torch
import torch.nn as nn



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



if __name__ == "__main__":
    class SimpleNet(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, output_size: int):
            super().__init__()
            self.embedding = nn.Embedding(19, 100)
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.layer2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            return self.layer2(x)

    # Initialize network with different strategies
    model = SimpleNet(10, 20, 2)

    # Using xavier initialization with ReLU
    xavier_init = get_initializer('xavier', activation='relu', embedding_init="normal")
    model.apply(xavier_init)



