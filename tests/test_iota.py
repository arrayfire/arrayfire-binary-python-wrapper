import random

import numpy as np
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
def test_iota_shape(shape: tuple) -> None:
    """Test if identity creates an array with the correct shape"""
    dtype = dtypes.s16
    t_shape = (1, 1)

    result = wrapper.iota(shape, t_shape, dtype)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203


def test_iota_invalid_shape() -> None:
    """Test if iota handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError) as excinfo:
        invalid_shape = (
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
        )
        dtype = dtypes.s16
        t_shape = ()

        wrapper.iota(invalid_shape, t_shape, dtype)

    assert f"CShape.__init__() takes from 1 to 5 positional arguments but {len(invalid_shape) + 1} were given" in str(
        excinfo.value
    )


@pytest.mark.parametrize(
    "t_shape",
    [
        (1,),
        (random.randint(1, 10), 1),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
def test_iota_tshape(t_shape: tuple) -> None:
    """Test if iota properly uses t_shape to change the size of the array and result in correct dimensions"""
    shape = np.array([2, 2])
    dtype = dtypes.s64

    if len(shape.shape) < len(t_shape):
        shape = np.append(shape, np.ones(len(t_shape) - len(shape), dtype=int))

    result_shape = shape * t_shape

    result = wrapper.iota(tuple(shape), t_shape, dtype)

    result_dims = tuple(int(value) for value in wrapper.get_dims(result))

    assert (result_dims[0 : len(result_shape)] == result_shape).all()  # noqa: E203


@pytest.mark.parametrize(
    "t_shape",
    [
        (0,),
        (-1, -1),
    ],
)
def test_iota_tshape_zero(t_shape: tuple) -> None:
    """Test it iota properly handles negative or zero t_shapes"""
    with pytest.raises(RuntimeError):
        shape = (2, 2)

        dtype = dtypes.s16

        wrapper.iota(shape, t_shape, dtype)


def test_iota_tshape_invalid() -> None:
    """Test it iota properly handles a tshape with greater than 4 dimensions"""
    with pytest.raises(TypeError):
        shape = (2, 2)
        invalid_tshape = (
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
        )
        dtype = dtypes.s16

        wrapper.iota(shape, invalid_tshape, dtype)


@pytest.mark.parametrize(
    "dtype_index",
    [i for i in range(13)],
)
def test_iota_dtype(dtype_index: int) -> None:
    """Test if iota creates an array with the correct dtype"""
    if (dtype_index in [1, 4]) or (dtype_index in [2, 3] and not wrapper.get_dbl_support()):
        pytest.skip()

    shape = (5, 5)
    t_shape = (2, 2)
    dtype = dtypes.c_api_value_to_dtype(dtype_index)

    result = wrapper.iota(shape, t_shape, dtype)

    assert dtypes.c_api_value_to_dtype(wrapper.get_type(result)) == dtype
