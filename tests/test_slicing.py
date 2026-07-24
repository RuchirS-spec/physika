import pytest
from tests.conftest import exec_phyk


@pytest.fixture(scope="module")
def slicing_ns():
    """
    Execute example_slicing.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("example_slicing")


class TestSlicing:
    """Test suites for ``examples/example_slicing.phyk file"""

    def test_program_level(self, slicing_ns):
        """Tests for program level correctness"""
        x_slice_1_to_3 = slicing_ns["x_slice_1_to_3"]
        assert x_slice_1_to_3.shape == (2, )
        assert x_slice_1_to_3.tolist() == [2, 3]

        x_slice_start_to_4 = slicing_ns["x_slice_start_to_4"]
        assert x_slice_start_to_4.shape == (4, )
        assert x_slice_start_to_4.tolist() == [1, 2, 3, 4]

        y_rows_1_to_3 = slicing_ns["y_rows_1_to_3"]
        assert y_rows_1_to_3.shape == (2, 4)
        assert y_rows_1_to_3.tolist() == [[5, 6, 7, 8], [9, 10, 11, 12]]

        y_column_2 = slicing_ns["y_column_2"]
        assert y_column_2.shape == (4, )
        assert y_column_2.tolist() == [3, 7, 11, 15]

        z_layer_1 = slicing_ns["z_layer_1"]
        assert z_layer_1.shape == (2, 2)
        assert z_layer_1.tolist() == [[5, 6], [7, 8]]

        z_first_element_each_layer = slicing_ns["z_first_element_each_layer"]
        assert len(z_first_element_each_layer) == 2
        assert z_first_element_each_layer.tolist() == [1, 5]

    def test_function_level(self, slicing_ns):
        """Tests for function level correctness"""
        func_x = slicing_ns["func_x"]
        assert func_x.shape == (2, )
        assert func_x.tolist() == [1, 2]

        func_y = slicing_ns["func_y"]
        assert func_y.shape == (4, )
        assert func_y.tolist() == [1, 5, 9, 13]

        func_z = slicing_ns["func_z"]
        assert func_z.shape == (2, )
        assert func_z.tolist() == [1, 5]

    def test_class_level(self, slicing_ns):
        """Tests for correctness inside class"""
        class_x = slicing_ns["class_x"]
        assert class_x.shape == (4, )
        assert class_x.tolist() == [1, 2, 3, 4]

        class_y = slicing_ns["class_y"]
        assert class_y.shape == (4, )
        assert class_y.tolist() == [2, 6, 10, 14]

        class_z = slicing_ns["class_z"]
        assert class_z.shape == (2, )
        assert class_z.tolist() == [3, 7]

    def test_slice_assignment(self, slicing_ns):
        """Tests for correctness of slice assignment"""
        x_assign = slicing_ns["x_assign"]
        assert x_assign.shape == (8, )
        assert x_assign.tolist() == [1, 10, 20, 4, 5, 6, 7, 8]

        y_assign = slicing_ns["y_assign"]
        assert y_assign.shape == (4, 4)
        assert y_assign.tolist() == [[10, 2, 3, 4], [20, 6, 7, 8],
                                     [30, 10, 11, 12], [40, 14, 15, 16]]

        z_assign = slicing_ns["z_assign"]
        assert z_assign.shape == (2, 2, 2)
        assert z_assign.tolist() == [[[10, 2], [20, 4]], [[10, 6], [20, 8]]]

    def test_slicing_inside_loops(self, slicing_ns):
        """Tests for correctness inside loops"""
        loop_y = slicing_ns["loop_y"]
        assert loop_y.shape == (4, 4)
        assert loop_y.tolist() == [[1, 2, 3, 4], [2, 6, 7, 8], [3, 10, 11, 12],
                                   [4, 14, 15, 16]]

        loop_z = slicing_ns["loop_z"]
        assert loop_z.shape == (2, 2, 2)
        assert loop_z.tolist() == [[[9, 2], [3, 4]], [[10, 6], [7, 8]]]
