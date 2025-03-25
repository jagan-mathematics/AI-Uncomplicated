from itertools import chain
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
from torch.distributed._tensor import DTensor


torch.autograd.detect_anomaly(True)

def validate_initialization(model: nn.Module, layer_type: str = None) -> Dict[str, Any]:
    """
    Test weight initialization properties of a model.

    Args:
        model: PyTorch model to test
        layer_type: Optional filter for specific layer type (e.g., 'Linear', 'Conv2d', 'Embedding')

    Returns:
        Dict containing test results
    """
    stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'near_zero': [],
        'layer_names': []
    }

    def analyze_weights(module: nn.Module, name: str):
        if hasattr(module, 'weight'):
            if layer_type is None or module.__class__.__name__ == layer_type:
                weights = module.weight.data.cpu().numpy()
                stats['mean'].append(np.mean(weights))
                stats['std'].append(np.std(weights))
                stats['min'].append(np.min(weights))
                stats['max'].append(np.max(weights))
                stats['near_zero'].append(
                    np.mean(np.abs(weights) < 1e-6)
                )
                stats['layer_names'].append(name)

    # Analyze each layer
    for name, module in model.named_modules():
        analyze_weights(module, name)

    return stats



def validate_embedding_padding(model: nn.Module, padding_idx: int = None) -> bool:
    """
    Test if embedding padding is correctly initialized to zero.
    """
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            if padding_idx is not None:
                padding_weights = module.weight.data[padding_idx].cpu().numpy()
                return np.allclose(padding_weights, 0)
    return False



def validate_activation_variance(model: nn.Module, input_size: tuple, n_samples: int = 1000):
    """
    Test if the variance of activations is maintained across layers.
    """
    model.eval()
    activations = {}

    # Hook to capture activations
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook


    def nan_wrap(name):
        def detect_nan_hook(module, input, output):
            """Hook to detect NaN values in module outputs."""
            if torch.isnan(output).any():
                print(f"NaN detected in {name}")
        return detect_nan_hook

    # Register hooks for each layer
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(hook_fn(name)))
            module.register_forward_hook(nan_wrap(name))

    # Generate random input
    x = torch.randn(n_samples, *input_size)
    x = x.long()
    x = torch.abs(x)

    # Forward pass
    with torch.no_grad():
        model(x.to(model.device))

    # Compute variance of activations
    variances = {name: act.var().item() for name, act in activations.items()}

    # Clean up hooks
    for handle in handles:
        handle.remove()

    return variances


def validate_model_initial_states(model, config, input_size, n_samples):
    stats = validate_initialization(model=model)

    print("\nInitialization Statistics:")
    for i, layer in enumerate(stats['layer_names']):
        print(f"\n{layer}:")
        print(f"Mean: {stats['mean'][i]:.6f}")
        print(f"Std:  {stats['std'][i]:.6f}")
        print(f"Min:  {stats['min'][i]:.6f}")
        print(f"Max:  {stats['max'][i]:.6f}")
        print(f"Near zero: {stats['near_zero'][i]:.2%}")

    # assert validate_embedding_padding(model=model, padding_idx=config.padding_id), "Validation of padding index failed"

    variances = validate_activation_variance(model=model, input_size=input_size, n_samples=n_samples)

    print("\nActivation Variances:")
    for layer, var in variances.items():
        print(f"{layer}: {var:.6f}")



@torch.no_grad()
def check_model_value_range(
    model: torch.nn.Module, range: float = 1e3, std: float = 1e3
):
    for name, param in chain(model.named_parameters(), model.named_buffers()):
        if isinstance(param, DTensor):
            param = param.to_local()

        if param.numel() == 0:
            print(f"WARNING: Model parameter {name} is empty, probably because of FSDP sharding")
            continue

        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"WARNING: Model parameter {name} contains NaN or Inf")

        param_range = param.max() - param.min()
        param_std = param.std()

        if param_range > range:
            print(
                f"WARNING: Model parameter {name} has a suspiciously large range ({param_range}): please check initialization and init_weights is defined and called"
            )
        if param_std > std:
            print(
                f"WARNING: Model parameter {name} has a suspiciously large standard deviation ({param_std}): please check initialization and init_weights is defined and called"
            )
        if (param == 0).all():
            print(
                f"WARNING: Model parameter {name} is all zeros: it might be because of a missing initialization"
            )


if __name__ == "__main__":
    from core.trainer.utils import get_initializer
    from core.models.GI_01.main.model import ConstrueAutoRegressiveModel
    from core.configurations.base import BaseConfiguration

    config = BaseConfiguration(model_name="small_lm", num_layers=6, hidden_dim=32, intermediate_dim=512,
                                max_positions=256, vocabulary_size=64000, num_heads=8, attention_dropout=0.05,
                                batch_size=8, weight_decay=0.01,
                                learning_rate=5e-4,
                                tokenizer_path="/workspace/vipin_g6/personal/pretraining/english_tokenizer/english_tokenizer.model",
                                dataset_batch_size=16, dataset_shuffle=True, num_epochs=2, eval_frequency=1,
                                eval_iter=10,
                                model_max_sequence=256)

    model = ConstrueAutoRegressiveModel(config=config)
    initalizer = get_initializer(init_type="xavier", activation="gelu", embedding_init="xavier", embedding_padding_idx=config.padding_id)

    model.apply(initalizer)
    validate_model_initial_states(model, config, (10, ), n_samples=1000)