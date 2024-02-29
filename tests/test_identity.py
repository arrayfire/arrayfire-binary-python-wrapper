import random

import pytest

import arrayfire_wrapper.dtypes as dtypes
import arrayfire_wrapper.lib as wrapper


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10), 1),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
def test_identity_shape(shape: tuple) -> None:
    """Test if identity creates an array with the correct shape"""
    dtype = dtypes.s16

    result = wrapper.identity(shape, dtype)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203


def test_identity_invalid_shape() -> None:
    """Test if identity handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError) as excinfo:
        invalid_shape = (
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
        )
        dtype = dtypes.s16

        wrapper.identity(invalid_shape, dtype)

    assert f"CShape.__init__() takes from 1 to 5 positional arguments but {len(invalid_shape) + 1} were given" in str(
        excinfo.value
    )


def test_identity_nonsquare_shape() -> None:
    dtype = dtypes.s16
    shape = (5, 6)

    result = wrapper.identity(shape, dtype)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203


@pytest.mark.parametrize(
    "dtype_index",
    [i for i in range(13)],
)
def test_identity_dtype(dtype_index: int) -> None:
    """Test if identity creates an array with the correct dtype"""
    if dtype_index in [2, 3] and not wrapper.get_dbl_support():
        pytest.skip()

    shape = (5, 5)
    dtype = dtypes.c_api_value_to_dtype(dtype_index)

    result = wrapper.identity(shape, dtype)

    assert dtypes.c_api_value_to_dtype(wrapper.get_type(result)) == dtype
