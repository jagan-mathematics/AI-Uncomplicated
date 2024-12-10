"""Test Decoder Module implementation"""
import torch
import unittest

from core.models.decoder import DecoderLayer


class TestDecoderLayer(unittest.TestCase):

    def setUp(self):
        self.model_dim = 512
        self.ffn_hidden = 2048
        self.num_heads = 8
        self.dropout = 0.1

        self.decoder_layer = DecoderLayer(
            model_dim=self.model_dim,
            ffn_hidden=self.ffn_hidden,
            num_head=self.num_heads,
            dropout=self.dropout
        )
        self.batch_size = 2
        self.sequence_length = 10

        self.sample_input = torch.rand(self.batch_size, self.sequence_length, self.model_dim)

        self.target_mask = torch.ones(
            self.batch_size,
            self.sequence_length,
            dtype=torch.bool
        )


    def test_layer_norm_effectiveness(self): # layer norm should not give tensors as null or too large values

        output = self.decoder_layer(
            self.sample_input,
            self.target_mask
        )

        self.assertFalse(torch.isnan(output).any()) # checking if any tensors is Nan
        self.assertTrue(output.max().item() > 10, "OUTPUT MAX ITEM IS GREATER THAN 10, so it is unstable")


    def test_shape(self):

        out = self.decoder_layer(
            self.sample_input,
            self.target_mask
        )

        self.assertEqual(
            out.shape,
            self.sample_input.shape
        )
    # asserting the input shape and output shape to be same


    def test_decoder_layer_attn(self):

        out = self.decoder_layer(
            self.sample_input,
            self.target_mask
        )

        print(f"out is {out}")

        # means 0 th token should not attend to 1st token

        for i in range(self.sequence_length):
            for j in range(i + 1, self.sequence_length):
                self.assertTrue((out[:, i, :] == out[:, j, :]).all(),
                                f"Token {i} should not attend to token {j} in a causal setup.")

