import pytest

import arrayfire_wrapper.dtypes as dtypes
import arrayfire_wrapper.lib as wrapper


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (3, 3, 3),
        (3, 3, 3, 3),
    ],
)
def test_diag_is_unit(shape: tuple) -> None:
    """Test if when is_unit_diag in lower returns an array with a unit diagonal"""
    dtype = dtypes.s64
    constant_array = wrapper.constant(3, shape, dtype)

    lower_array = wrapper.upper(constant_array, True)
    diagonal = wrapper.diag_extract(lower_array, 0)
    diagonal_value = wrapper.get_scalar(diagonal, dtype)

    assert diagonal_value == 1


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (3, 3, 3),
        (3, 3, 3, 3),
    ],
)
def test_is_original(shape: tuple) -> None:
    """Test if is_original keeps the diagonal the same as the original array"""
    dtype = dtypes.s64
    constant_array = wrapper.constant(3, shape, dtype)
    original_value = wrapper.get_scalar(constant_array, dtype)

    lower_array = wrapper.upper(constant_array, False)
    diagonal = wrapper.diag_extract(lower_array, 0)
    diagonal_value = wrapper.get_scalar(diagonal, dtype)

    assert original_value == diagonal_value
