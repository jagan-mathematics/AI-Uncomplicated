import torch
from core.utils.masks import _update_causal_mask


def testSingleTokenValue():
    input_tensor = torch.rand(2, 1, 1, 1)
    attention_mask = torch.as_tensor([[1], [0]])
    
    mask = _update_causal_mask(input_tensor=input_tensor, attention_mask=attention_mask)
    assert torch._is_all_true(mask[0][0] == -0.0)
    assert torch._is_all_true(mask[1][0] < 0.0)
    
    
    
def testTestMultiHeadMasking():
    input_tensor = torch.rand(2, 2, 3, 3)
    attention_mask = torch.as_tensor([[1, 1, 0], [1, 0, 0]])
    
    mask = _update_causal_mask(input_tensor=input_tensor, attention_mask=attention_mask)

    batch_zero_expected = torch.as_tensor([[False, True, True],
                                          [False, False, True],
                                          [False, False, True]])

    batch_one_expected = torch.as_tensor([[False, True, True],
                                          [False, True, True],
                                          [False, True, True]])

    assert torch._is_all_true((mask[0][0] < 0.0) == batch_zero_expected)
    assert torch._is_all_true((mask[1][0] < 0.0) == batch_one_expected)
    
if __name__ == "__main__":
    testSingleTokenValue()
    testTestMultiHeadMasking()
