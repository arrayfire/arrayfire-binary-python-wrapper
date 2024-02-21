import random

import pytest

import arrayfire_wrapper.dtypes as dtypes
from arrayfire_wrapper.lib.create_and_modify_array import manage_array
from arrayfire_wrapper.lib.create_and_modify_array.create_array import (
    constant,
    constant_complex,
    constant_long,
    constant_ulong,
)
from arrayfire_wrapper.lib.create_and_modify_array.create_array.random_number_generation import randu
from arrayfire_wrapper.lib.create_and_modify_array.manage_device import get_dbl_support

invalid_shape = (
    random.randint(1, 10),
    random.randint(1, 10),
    random.randint(1, 10),
    random.randint(1, 10),
    random.randint(1, 10),
)


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
def test_constant_shape(shape: tuple) -> None:
    """Test if constant creates an array with the correct shape."""
    number = 5.0
    dtype = dtypes.s16

    result = constant(number, shape, dtype)

    assert manage_array.get_dims(result)[0:len(shape)] == shape


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
def test_constant_complex_shape(shape: tuple) -> None:
    """Test if constant_complex creates an array with the correct shape."""
    dtype = dtypes.c32
    rand_array = randu((1, 1), dtype)
    number = manage_array.get_scalar(rand_array, dtype)

    if isinstance(number, (int, float, complex)):
        result = constant_complex(number, shape, dtype)

        assert manage_array.get_dims(result)[0:len(shape)] == shape
    else:
        pytest.skip()


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
def test_constant_long_shape(shape: tuple) -> None:
    """Test if constant_long creates an array with the correct shape."""
    dtype = dtypes.s64
    rand_array = randu((1, 1), dtype)
    number = manage_array.get_scalar(rand_array, dtype)

    if isinstance(number, (int, float)):
        result = constant_long(number, shape, dtype)

        assert manage_array.get_dims(result)[0:len(shape)] == shape


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
def test_constant_ulong_shape(shape: tuple) -> None:
    """Test if constant_ulong creates an array with the correct shape."""
    dtype = dtypes.u64
    rand_array = randu((1, 1), dtype)
    number = manage_array.get_scalar(rand_array, dtype)

    if isinstance(number, (int, float)):
        result = constant_ulong(number, shape, dtype)

        assert manage_array.get_dims(result)[0:len(shape)] == shape
    else:
        pytest.skip()


def test_constant_shape_invalid() -> None:
    """Test if constant handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError) as excinfo:
        number = 5.0
        dtype = dtypes.s16

        constant(number, invalid_shape, dtype)

    assert f"CShape.__init__() takes from 1 to 5 positional arguments but {len(invalid_shape) + 1} were given" in str(
        excinfo.value
    )


def test_constant_complex_shape_invalid() -> None:
    """Test if constant_complex handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError) as excinfo:
        dtype = dtypes.c32
        rand_array = randu((1, 1), dtype)
        number = manage_array.get_scalar(rand_array, dtype)

        if isinstance(number, (int, float, complex)):
            constant_complex(number, invalid_shape, dtype)

    assert f"CShape.__init__() takes from 1 to 5 positional arguments but {len(invalid_shape) + 1} were given" in str(
        excinfo.value
    )


def test_constant_long_shape_invalid() -> None:
    """Test if constant_long handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError) as excinfo:
        dtype = dtypes.s64
        rand_array = randu((1, 1), dtype)
        number = manage_array.get_scalar(rand_array, dtype)

        if isinstance(number, (int, float)):
            constant_long(number, invalid_shape, dtype)

    assert f"CShape.__init__() takes from 1 to 5 positional arguments but {len(invalid_shape) + 1} were given" in str(
        excinfo.value
    )


def test_constant_ulong_shape_invalid() -> None:
    """Test if constant_ulong handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError) as excinfo:
        dtype = dtypes.u64
        rand_array = randu((1, 1), dtype)
        number = manage_array.get_scalar(rand_array, dtype)

        if isinstance(number, (int, float)):
            constant_ulong(number, invalid_shape, dtype)

    assert f"CShape.__init__() takes from 1 to 5 positional arguments but {len(invalid_shape) + 1} were given" in str(
        excinfo.value
    )


@pytest.mark.parametrize(
    "dtype_index",
    [i for i in range(13)],
)
def test_constant_dtype(dtype_index: int) -> None:
    """Test if constant creates an array with the correct dtype."""
    if dtype_index in [1, 3] or (dtype_index == 2 and not get_dbl_support()):
        pytest.skip()

    dtype = dtypes.c_api_value_to_dtype(dtype_index)

    rand_array = randu((1, 1), dtype)
    value = manage_array.get_scalar(rand_array, dtype)
    shape = (2, 2)
    if isinstance(value, (int, float)):
        result = constant(value, shape, dtype)
        assert dtypes.c_api_value_to_dtype(manage_array.get_type(result)) == dtype
    else:
        pytest.skip()


@pytest.mark.parametrize(
    "dtype_index",
    [i for i in range(13)],
)
def test_constant_complex_dtype(dtype_index: int) -> None:
    """Test if constant_complex creates an array with the correct dtype."""
    if dtype_index not in [1, 3] or (dtype_index == 3 and not get_dbl_support()):
        pytest.skip()

    dtype = dtypes.c_api_value_to_dtype(dtype_index)
    rand_array = randu((1, 1), dtype)
    value = manage_array.get_scalar(rand_array, dtype)
    shape = (2, 2)

    if isinstance(value, (int, float, complex)):
        result = constant_complex(value, shape, dtype)

        assert dtypes.c_api_value_to_dtype(manage_array.get_type(result)) == dtype
    else:
        pytest.skip()


def test_constant_long_dtype() -> None:
    """Test if constant_long creates an array with the correct dtype."""
    dtype = dtypes.s64

    rand_array = randu((1, 1), dtype)
    value = manage_array.get_scalar(rand_array, dtype)
    shape = (2, 2)

    if isinstance(value, (int, float)):
        result = constant_long(value, shape, dtype)

        assert dtypes.c_api_value_to_dtype(manage_array.get_type(result)) == dtype
    else:
        pytest.skip()


def test_constant_ulong_dtype() -> None:
    """Test if constant_ulong creates an array with the correct dtype."""
    dtype = dtypes.u64

    rand_array = randu((1, 1), dtype)
    value = manage_array.get_scalar(rand_array, dtype)
    shape = (2, 2)

    if isinstance(value, (int, float)):
        result = constant_ulong(value, shape, dtype)

        assert dtypes.c_api_value_to_dtype(manage_array.get_type(result)) == dtype
    else:
        pytest.skip()
