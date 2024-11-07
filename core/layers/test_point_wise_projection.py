import torch

from core.layers.point_wise_projection import PointWiseProjection
from core.layers.point_wise_projection import PointWiseGatedProjection
from core.configurations.base import BaseConfiguration



# Mock BaseConfiguration for testing
class MockConfiguration(BaseConfiguration):
    def __init__(self, hidden_dim, intermediate_dim):
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim



def test_pointwise_projection(mock_config, input_tensor):
    model = PointWiseProjection(config=mock_config)
    
    # Ensure the model can process input without errors
    output = model(input_tensor)
    
    # Check the output shape
    assert output.shape == torch.Size([4, 8]), "Output shape is incorrect"
    
    # Check that output is not NaN or Inf
    assert torch.all(torch.isfinite(output)), "Output contains NaN or Inf values"

def test_pointwise_gated_projection(mock_config, input_tensor):
    model = PointWiseGatedProjection(config=mock_config)
    
    # Ensure the model can process input without errors
    output = model(input_tensor)
    
    # Check the output shape
    assert output.shape == torch.Size([4, 8]), "Output shape is incorrect"
    
    # Check that output is not NaN or Inf
    assert torch.all(torch.isfinite(output)), "Output contains NaN or Inf values"


def test_weight_initialization(mock_config):
    model = PointWiseProjection(config=mock_config)
    # Check if weights are initialized correctly
    assert model.up_projection.weight is not None, "Up projection weights are not initialized"
    assert model.down_projection.weight is not None, "Down projection weights are not initialized"

    model_gated = PointWiseGatedProjection(config=mock_config)
    assert model_gated.gate_projection.weight is not None, "Gate projection weights are not initialized"
    assert model_gated.up_projection.weight is not None, "Up projection weights are not initialized"
    assert model_gated.down_projection.weight is not None, "Down projection weights are not initialized"



if __name__ == "__main__":
    mock_config = MockConfiguration(hidden_dim=8, intermediate_dim=16)
    input_tensor = torch.rand((4, 8))
    
    test_pointwise_projection(mock_config=mock_config, input_tensor=input_tensor)
    test_pointwise_gated_projection(mock_config=mock_config, input_tensor=input_tensor)
    test_weight_initialization(mock_config=mock_config)
