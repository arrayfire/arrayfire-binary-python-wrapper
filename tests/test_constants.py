import random

import pytest

import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.dtypes import (
    Dtype,
    c32,
    c64,
    c_api_value_to_dtype,
    f16,
    f32,
    f64,
    s16,
    s32,
    s64,
    u8,
    u16,
    u32,
    u64,
)

invalid_shape = (
    random.randint(1, 10),
    random.randint(1, 10),
    random.randint(1, 10),
    random.randint(1, 10),
    random.randint(1, 10),
)


types = [s16, s32, s64, u8, u16, u32, u64, f16, f32, f64, c32, c64]


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
    dtype = s16

    result = wrapper.constant(number, shape, dtype)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203


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
    dtype = c32

    dtype = c32
    rand_array = wrapper.randu((1, 1), dtype)
    number = wrapper.get_scalar(rand_array, dtype)

    if isinstance(number, (complex)):
        result = wrapper.constant_complex(number, shape, dtype)
        assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203
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
    dtype = s64
    rand_array = wrapper.randu((1, 1), dtype)
    number = wrapper.get_scalar(rand_array, dtype)

    if isinstance(number, (int, float)):
        result = wrapper.constant_long(number, shape, dtype)

        assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203


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
    dtype = u64
    rand_array = wrapper.randu((1, 1), dtype)
    number = wrapper.get_scalar(rand_array, dtype)

    if isinstance(number, (int, float)):
        result = wrapper.constant_ulong(number, shape, dtype)

        assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203
    else:
        pytest.skip()


def test_constant_shape_invalid() -> None:
    """Test if constant handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError):
        number = 5.0
        dtype = s16

        wrapper.constant(number, invalid_shape, dtype)


def test_constant_complex_shape_invalid() -> None:
    """Test if constant_complex handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError):
        dtype = c32
        rand_array = wrapper.randu((1, 1), dtype)
        number = wrapper.get_scalar(rand_array, dtype)

        if isinstance(number, (int, float, complex)):
            wrapper.constant_complex(number, invalid_shape, dtype)


def test_constant_long_shape_invalid() -> None:
    """Test if constant_long handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError):
        dtype = s64
        rand_array = wrapper.randu((1, 1), dtype)
        number = wrapper.get_scalar(rand_array, dtype)

        if isinstance(number, (int, float)):
            wrapper.constant_long(number, invalid_shape, dtype)


def test_constant_ulong_shape_invalid() -> None:
    """Test if constant_ulong handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError):
        dtype = u64
        rand_array = wrapper.randu((1, 1), dtype)
        number = wrapper.get_scalar(rand_array, dtype)

        if isinstance(number, (int, float)):
            wrapper.constant_ulong(number, invalid_shape, dtype)


@pytest.mark.parametrize(
    "dtype",
    types,
)
def test_constant_dtype(dtype: Dtype) -> None:
    """Test if constant creates an array with the correct dtype."""
    if dtype in [c32, c64] or (dtype == f64 and not wrapper.get_dbl_support()):
        pytest.skip()

    rand_array = wrapper.randu((1, 1), dtype)
    value = wrapper.get_scalar(rand_array, dtype)
    shape = (2, 2)
    if isinstance(value, (int, float)):
        result = wrapper.constant(value, shape, dtype)
        assert c_api_value_to_dtype(wrapper.get_type(result)) == dtype
    else:
        pytest.skip()


@pytest.mark.parametrize(
    "dtype",
    types,
)
def test_constant_complex_dtype(dtype: Dtype) -> None:
    """Test if constant_complex creates an array with the correct dtype."""
    if dtype not in [c32, c64] or (dtype == c64 and not wrapper.get_dbl_support()):
        pytest.skip()

    rand_array = wrapper.randu((1, 1), dtype)
    value = wrapper.get_scalar(rand_array, dtype)
    shape = (2, 2)

    if isinstance(value, (int, float, complex)):
        result = wrapper.constant_complex(value, shape, dtype)
        assert c_api_value_to_dtype(wrapper.get_type(result)) == dtype
    else:
        pytest.skip()


def test_constant_long_dtype() -> None:
    """Test if constant_long creates an array with the correct dtype."""
    dtype = s64

    rand_array = wrapper.randu((1, 1), dtype)
    value = wrapper.get_scalar(rand_array, dtype)
    shape = (2, 2)

    if isinstance(value, (int, float)):
        result = wrapper.constant_long(value, shape, dtype)

        assert c_api_value_to_dtype(wrapper.get_type(result)) == dtype
    else:
        pytest.skip()


def test_constant_ulong_dtype() -> None:
    """Test if constant_ulong creates an array with the correct dtype."""
    dtype = u64

    rand_array = wrapper.randu((1, 1), dtype)
    value = wrapper.get_scalar(rand_array, dtype)
    shape = (2, 2)

    if isinstance(value, (int, float)):
        result = wrapper.constant_ulong(value, shape, dtype)

        assert c_api_value_to_dtype(wrapper.get_type(result)) == dtype
    else:
        pytest.skip()
