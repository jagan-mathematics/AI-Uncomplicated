from torch import nn, Tensor


class PytorchGELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def forward(self, input_tensor: Tensor) -> Tensor:
        return nn.functional.gelu(input_tensor, approximate="tanh")