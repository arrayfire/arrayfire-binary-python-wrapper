import random

import pytest

import arrayfire_wrapper.dtypes as dtypes
import arrayfire_wrapper.lib as wrapper

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

    result = wrapper.constant(number, shape, dtype)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203


# @pytest.mark.parametrize(
#     "shape",
#     [
#         (),
#         (random.randint(1, 10), 1),
#         (random.randint(1, 10), random.randint(1, 10)),
#         (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
#         (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
#     ],
# )
# # def test_constant_complex_shape(shape: tuple) -> None:
# #     """Test if constant_complex creates an array with the correct shape."""
# #     dtype = dtypes.c32
# #     rand_array = wrapper.randu((1, 1), dtype)
# #     number = wrapper.get_scalar(rand_array, dtype)

# #     if isinstance(number, (int, float, complex)):
# #         result = wrapper.constant_complex(number, shape, dtype)
# #         assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203
# #     else:
# #         pytest.skip()


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
    dtype = dtypes.u64
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
        dtype = dtypes.s16

        wrapper.constant(number, invalid_shape, dtype)


# def test_constant_complex_shape_invalid() -> None:
#     """Test if constant_complex handles a shape with greater than 4 dimensions"""
#     with pytest.raises(TypeError):
#         dtype = dtypes.c32
#         rand_array = wrapper.randu((1, 1), dtype)
#         number = wrapper.get_scalar(rand_array, dtype)

#         if isinstance(number, (int, float, complex)):
#             wrapper.constant_complex(number, invalid_shape, dtype)


def test_constant_long_shape_invalid() -> None:
    """Test if constant_long handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError):
        dtype = dtypes.s64
        rand_array = wrapper.randu((1, 1), dtype)
        number = wrapper.get_scalar(rand_array, dtype)

        if isinstance(number, (int, float)):
            wrapper.constant_long(number, invalid_shape, dtype)


def test_constant_ulong_shape_invalid() -> None:
    """Test if constant_ulong handles a shape with greater than 4 dimensions"""
    with pytest.raises(TypeError):
        dtype = dtypes.u64
        rand_array = wrapper.randu((1, 1), dtype)
        number = wrapper.get_scalar(rand_array, dtype)

        if isinstance(number, (int, float)):
            wrapper.constant_ulong(number, invalid_shape, dtype)


@pytest.mark.parametrize(
    "dtype_index",
    [i for i in range(13)],
)
def test_constant_dtype(dtype_index: int) -> None:
    """Test if constant creates an array with the correct dtype."""
    if dtype_index in [1, 3] or (dtype_index == 2 and not wrapper.get_dbl_support()):
        pytest.skip()

    dtype = dtypes.c_api_value_to_dtype(dtype_index)

    rand_array = wrapper.randu((1, 1), dtype)
    value = wrapper.get_scalar(rand_array, dtype)
    shape = (2, 2)
    if isinstance(value, (int, float)):
        result = wrapper.constant(value, shape, dtype)
        assert dtypes.c_api_value_to_dtype(wrapper.get_type(result)) == dtype
    else:
        pytest.skip()


# @pytest.mark.parametrize(
#     "dtype_index",
#     [i for i in range(13)],
# )
# def test_constant_complex_dtype(dtype_index: int) -> None:
#     """Test if constant_complex creates an array with the correct dtype."""
#     if dtype_index not in [1, 3] or (dtype_index == 3 and not wrapper.get_dbl_support()):
#         pytest.skip()

#     dtype = dtypes.c_api_value_to_dtype(dtype_index)
#     rand_array = wrapper.randu((1, 1), dtype)
#     value = wrapper.get_scalar(rand_array, dtype)
#     shape = (2, 2)

#     if isinstance(value, (int, float, complex)):
#         result = wrapper.constant_complex(value, shape, dtype)

#         assert dtypes.c_api_value_to_dtype(wrapper.get_type(result)) == dtype
#     else:
#         pytest.skip()


def test_constant_long_dtype() -> None:
    """Test if constant_long creates an array with the correct dtype."""
    dtype = dtypes.s64

    rand_array = wrapper.randu((1, 1), dtype)
    value = wrapper.get_scalar(rand_array, dtype)
    shape = (2, 2)

    if isinstance(value, (int, float)):
        result = wrapper.constant_long(value, shape, dtype)

        assert dtypes.c_api_value_to_dtype(wrapper.get_type(result)) == dtype
    else:
        pytest.skip()


def test_constant_ulong_dtype() -> None:
    """Test if constant_ulong creates an array with the correct dtype."""
    dtype = dtypes.u64

    rand_array = wrapper.randu((1, 1), dtype)
    value = wrapper.get_scalar(rand_array, dtype)
    shape = (2, 2)

    if isinstance(value, (int, float)):
        result = wrapper.constant_ulong(value, shape, dtype)

        assert dtypes.c_api_value_to_dtype(wrapper.get_type(result)) == dtype
    else:
        pytest.skip()
