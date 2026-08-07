import torch
from physika.runtime import random_complex, compl_mul1d


class TestRandomComplex:
    """Tests for ``random_complex`` function."""

    def test_output_is_complex(self):
        # Check that the returned tensor has a complex dtype.
        x = random_complex(4, 8)
        assert torch.is_complex(x)

    def test_output_shape(self):
        # Check that the output shape matches the requested shape.
        x = random_complex(3, 5, 7)
        assert x.shape == (3, 5, 7)

    def test_single_dim_shape(self):
        # Check that a single-dimension shape works correctly.
        x = random_complex(16)
        assert x.shape == (16, )


class TestComplMul1d:
    """Tests for compl_mul1d, the spectral-domain multiplication used in
    FNO's spectral convolution layer."""

    def test_output_shape(self):
        # Check that output channels/modes match the weights shape.
        x_ft = random_complex(4, 16)
        weights1 = random_complex(4, 8, 16)
        out = compl_mul1d(x_ft, weights1)
        assert out.shape == (8, 16)

    def test_output_is_complex(self):
        # Check that the result stays complex-valued after the einsum.
        x_ft = random_complex(2, 4)
        weights1 = random_complex(2, 3, 4)
        out = compl_mul1d(x_ft, weights1)
        assert torch.is_complex(out)

    def test_zero_weights_give_zero_output(self):
        # Check that zero spectral weights produce a zero output,
        # regardless of the input.
        x_ft = random_complex(3, 6)
        weights1 = torch.zeros(3, 5, 6, dtype=torch.cfloat)
        out = compl_mul1d(x_ft, weights1)
        assert torch.allclose(out, torch.zeros_like(out))
