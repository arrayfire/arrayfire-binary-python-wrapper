import pytest

import arrayfire_wrapper.dtypes as dtypes
import arrayfire_wrapper.lib as wrapper


@pytest.mark.parametrize("diagonal_shape", [(2,), (10,), (100,), (1000,)])
def test_diagonal_shape(diagonal_shape: tuple) -> None:
    """Test if diagonal array is keeping the shape of the passed into the input array"""
    in_arr = wrapper.constant(1, diagonal_shape, dtypes.s16)
    diag_array = wrapper.diag_create(in_arr, 0)

    extracted_diagonal = wrapper.diag_extract(diag_array, 0)

    assert wrapper.get_dims(extracted_diagonal)[0 : len(diagonal_shape)] == diagonal_shape  # noqa: E203


@pytest.mark.parametrize("diagonal_shape", [(2,), (10,), (100,), (1000,)])
def test_diagonal_val(diagonal_shape: tuple) -> None:
    """Test if diagonal array is keeping the same value as that of the values passed into the input array"""
    dtype = dtypes.s16
    in_arr = wrapper.constant(1, diagonal_shape, dtype)
    diag_array = wrapper.diag_create(in_arr, 0)

    extracted_diagonal = wrapper.diag_extract(diag_array, 0)

    assert wrapper.get_scalar(extracted_diagonal, dtype) == wrapper.get_scalar(in_arr, dtype)


@pytest.mark.parametrize(
    "diagonal_shape",
    [
        (10, 10, 10),
        (100, 100, 100, 100),
    ],
)
def test_invalid_diagonal(diagonal_shape: tuple) -> None:
    """Test if an invalid diagonal shape is being properly handled"""
    with pytest.raises(RuntimeError):
        in_arr = wrapper.constant(1, diagonal_shape, dtypes.s16)
        diag_array = wrapper.diag_create(in_arr, 0)

        wrapper.diag_extract(diag_array, 0)
