import torch
from physika.runtime import random_complex, compl_mul1d
from tests.conftest import type_errors, capture_output


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


class TestPhysikaPrint:
    """Tests for ``physika_print()`` function"""

    def test_physika_print_program_level(self):
        # Test ``physika_print()`` at program level.
        src = ("x: ℝ = 1\n"
               "physika_print(x)\n")
        assert type_errors(src) == []
        out = capture_output(src)
        assert "1" in out

    def test_physika_print_function_level(self):
        # Test ``physika_print()`` in function level.
        src = ("def test_f(x: ℝ): ℝ:\n"
               "    physika_print(x)\n"
               "    return 1\n"
               "test_f(10)\n")
        assert type_errors(src) == []
        out = capture_output(src)
        assert "10" in out

    def test_physika_print_inside_class(self):
        # Test ``physika_print()`` inside class.
        src = ("class Temp:\n"
               "    def f(x: ℝ): ℝ:\n"
               "        physika_print(x)\n"
               "        return 1\n"
               "obj: Temp = Temp()\n"
               "obj.f(10)\n")
        assert type_errors(src) == []
        out = capture_output(src)
        assert "10" in out

    def test_physika_print_inside_loop(self):
        # Test ``physika_print()`` inside loops.
        src = ("arr: ℝ[2] = [1, 2]\n"
               "for i: ℕ(2):\n"
               "    physika_print(arr[i])\n")
        assert type_errors(src) == []
        out = capture_output(src)
        assert "1" in out
        assert "2" in out
    
    def test_physika_print_inside_function_loop(self):
        # Test ``physika_print()`` inside function loops.
        src = ("def test_f(x: ℝ[m]): ℝ:\n"
               "    for k:ℕ(2):\n"
               "        physika_print(x[k])\n"
               "    return 1\n"
               "arr: ℝ[2] = [10, 20]\n"
               "test_f(arr)")
        assert type_errors(src) == []
        out = capture_output(src) == []
        out = capture_output(src)
        assert "10" in out
        assert "20" in out
        assert "1" in out
