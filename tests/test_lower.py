import pytest

import arrayfire_wrapper.dtypes as dtypes
from arrayfire_wrapper.lib.create_and_modify_array.create_array.constant import constant
from arrayfire_wrapper.lib.create_and_modify_array.create_array.diag import diag_extract
from arrayfire_wrapper.lib.create_and_modify_array.create_array.lower import lower
from arrayfire_wrapper.lib.create_and_modify_array.manage_array import get_scalar


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
    constant_array = constant(3, shape, dtype)

    lower_array = lower(constant_array, True)
    diagonal = diag_extract(lower_array, 0)
    diagonal_value = get_scalar(diagonal, dtype)

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
    constant_array = constant(3, shape, dtype)
    original_value = get_scalar(constant_array, dtype)

    lower_array = lower(constant_array, False)
    diagonal = diag_extract(lower_array, 0)
    diagonal_value = get_scalar(diagonal, dtype)

    assert original_value == diagonal_value
