"""Test case module for Layer Norm"""


# Main objective of this test case is to see whether the Layer Norm is correctly normalizing the tensors or not
# such that mean should be 0 and SD should be 1

import torch
import unittest

from core.layers.norms import LayerNorm


class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        self.layer_norm = LayerNorm(model_dimension=4)


    def test_batch_size_preservation(self): # test case to check dimension equals or not
        input_tensor = torch.randn(2, 4)
        output_tensor = self.layer_norm(input_tensor)
        self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_mean_and_variance_norm(self):
        input_tensor = torch.randn(10, 4)
        output_tensor = self.layer_norm(input_tensor)
        mean = output_tensor.mean(-1) # Checking Mean Value across model dimension
        variance = output_tensor.var(-1, unbiased=False) # checking Variance value across model dimension
        self.assertTrue(
            torch.allclose(mean, torch.zeros_like(mean), atol=1e-5) # mean should be somewhat close to 0
        )
        self.assertTrue(
            torch.allclose(variance, torch.ones_like(variance), atol=1e-5) # variance should be somewhat close to 1
        )

    def test_gamma_and_beta(self):
        self.layer_norm.gamma.data = torch.tensor([2.0, 2.0, 2.0, 2.0])
        self.layer_norm.beta.data = torch.tensor([1.0, 1.0, 1.0, 1.0])

        input_tensor = torch.randn(5, 4)
        output_tensor = self.layer_norm(input_tensor)

        # print(f"output tensor is {output_tensor[0]}")
        mean = input_tensor[0].mean()
        variance = input_tensor[0].var(unbiased=False)

        normalized_input = (
                                   input_tensor[0] - mean
                           ) / torch.sqrt(variance + self.layer_norm.epsilon)

        expected_output = self.layer_norm.gamma * normalized_input + self.layer_norm.beta

        self.assertTrue(
            torch.allclose(
                output_tensor[0], expected_output, atol=1e-5
            )
        )
