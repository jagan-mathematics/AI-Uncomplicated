from core.activations.gelu import PytorchGELUTanh
import torch
import math


def testPositiveRegions():
    activate = PytorchGELUTanh()
    
    values = [10, 100, 1.0, -0.122, -100]
    input_tensor = torch.as_tensor(values, dtype=torch.float32)
    expected_value = lambda x: (0.5 * x) * (1 + math.tanh((math.sqrt(2/math.pi) * (x + 0.044715 * x**3))))
    
    activation_output = activate(input_tensor)
    expected_activation = torch.as_tensor([expected_value(i) for i in values])    
    torch.testing.assert_close(activation_output, expected_activation, rtol=0.01, atol=0.001)
    
    
    

if __name__ == "__main__":
    testPositiveRegions()