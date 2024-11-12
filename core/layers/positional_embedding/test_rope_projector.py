from core.layers.positional_embedding.rope_projector import RopePositionEmbedding
from core.layers.positional_embedding.rope_projector import rotate_half, apply_positional_embedding

import torch


class TestRopePositionEmbedding:
    def __init__(self):
        self.test_initialization()
        self.test_forward_pass()
        self.test_device_compatibility()
        self.test_no_gradients()
        
        
    def test_initialization(self):
        hidden_dim = 16
        max_positions = 2048
        base = 10000
        
        model = RopePositionEmbedding(hidden_dim, max_positions, base)
        
        assert model.hidden_dim == hidden_dim
        assert model.base == base
        assert hasattr(model, "rotatory_matrix")
        assert model.rotatory_matrix.shape == (max_positions, hidden_dim // 2)
        assert not model.rotatory_matrix.requires_grad
    
    
    def test_forward_pass(self):
        hidden_dim = 16
        sequence_length = 10
        batch_size = 4

        model = RopePositionEmbedding(hidden_dim)
        x = torch.randn(batch_size, sequence_length, hidden_dim)
        
        cos, sin = model(x)
        
        # Ensure shapes match
        assert cos.shape == sin.shape == (batch_size, sequence_length, hidden_dim)
        # Ensure output dtype matches input dtype
        assert cos.dtype == x.dtype
        assert sin.dtype == x.dtype
        
    
    def test_device_compatibility(self):
        hidden_dim = 16
        sequence_length = 10
        batch_size = 4

        model = RopePositionEmbedding(hidden_dim)
        x_cpu = torch.randn(batch_size, sequence_length, hidden_dim)
        cos_cpu, sin_cpu = model(x_cpu)
        
        # Test on GPU if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            x_gpu = x_cpu.to('cuda')
            cos_gpu, sin_gpu = model(x_gpu)
            
            # Ensure shapes match
            assert cos_gpu.shape == sin_gpu.shape == (batch_size, sequence_length, hidden_dim)
            # Ensure values match between CPU and GPU outputs
            assert torch.allclose(cos_cpu, cos_gpu.cpu(), atol=1e-6)
            assert torch.allclose(sin_cpu, sin_gpu.cpu(), atol=1e-6)
    
    
    def test_no_gradients(self):
        hidden_dim = 16
        sequence_length = 10
        batch_size = 4

        model = RopePositionEmbedding(hidden_dim)
        x = torch.randn(batch_size, sequence_length, hidden_dim, requires_grad=True)

        with torch.no_grad():
            cos, sin = model(x)

        # Ensure gradients are not computed
        assert not cos.requires_grad
        assert not sin.requires_grad
        


class TestRotateHalf:
    def __init__(self):
        self.test_rotate_half()
        self.test_3d_tensor()
        self.test_single_batch()
        self.test_exceptioin_check()
    
    def test_rotate_half(self):
        x = torch.tensor([[1, 2, 3, 4],
                        [5, 6, 7, 8]], dtype=torch.float32)
        rotated = rotate_half(x)
        expected = torch.tensor([[-3, -4, 1, 2],
                                [-7, -8, 5, 6]], dtype=torch.float32)
        assert torch.allclose(rotated, expected)
        
    
    def test_single_batch(self):
        x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        rotated = rotate_half(x)
        expected = torch.tensor([-3, -4, 1, 2], dtype=torch.float32)
        assert torch.allclose(rotated, expected)
        
    
    def test_3d_tensor(self):
        x = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]],
                        [[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=torch.float32)
        rotated = rotate_half(x)
        expected = torch.tensor([[[-3, -4, 1, 2], [-7, -8, 5, 6]],
                                [[-11, -12, 9, 10], [-15, -16, 13, 14]]], dtype=torch.float32)
        assert torch.allclose(rotated, expected)
        
    
    def test_exceptioin_check(self):
        try:
            x = torch.tensor([1, 2, 3], dtype=torch.float32)
            rotate_half(x)
            assert False
        except AssertionError:
            pass  # Expected behavior
            


class TestApplyPositionalEmbedding:
    def __init__(self):
        self.test_apply_positional_embedding()
        self.test_calculation()
    
    def test_apply_positional_embedding(self):
        batch_size = 2
        num_heads = 4
        seq_length = 8
        dim = 16

        q = torch.randn(batch_size, num_heads, seq_length, dim)
        k = torch.randn(batch_size, num_heads, seq_length, dim)
        cos = torch.randn(batch_size, seq_length, dim)
        sin = torch.randn(batch_size, seq_length, dim)
        
        # Call the function
        q_out, k_out = apply_positional_embedding(q, k, cos, sin)
        
        # Verify shapes
        assert q_out.shape == (batch_size, num_heads, seq_length, dim), "Incorrect shape for q_out"
        assert k_out.shape == (batch_size, num_heads, seq_length, dim), "Incorrect shape for k_out"
        
    
    def test_calculation(self):
        batch_size = 2
        num_heads = 4
        seq_length = 8
        dim = 16
        
        # Check numerical behavior (sanity check)
        q = torch.ones(batch_size, num_heads, seq_length, dim)
        k = torch.ones(batch_size, num_heads, seq_length, dim)
        cos = torch.ones(batch_size, seq_length, dim)
        sin = torch.zeros(batch_size, seq_length, dim)  # Sinusoidal embedding disabled
        
        q_out, k_out = apply_positional_embedding(q, k, cos, sin)
        assert torch.allclose(q_out, q), "Output q should match input q when sin is zero"
        assert torch.allclose(k_out, k), "Output k should match input k when sin is zero"



if __name__ == "__main__":
    TestRopePositionEmbedding()
    TestRotateHalf()
    TestApplyPositionalEmbedding()
    print("Success.... All Pass 🥳..")
