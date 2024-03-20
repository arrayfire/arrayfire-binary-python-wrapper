import random

import numpy as np
import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper

# import arrayfire_wrapper.lib.mathematical_functions as ops

from . import utility_functions as util


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10),),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
def test_multiply_shapes(shape: tuple) -> None:
    """Test multiplication operation between two arrays of the same shape"""
    lhs = wrapper.randu(shape, dtype.f16)
    rhs = wrapper.randu(shape, dtype.f16)

    result = wrapper.mul(lhs, rhs)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa


def test_multiply_different_shapes() -> None:
    """Test if multiplication handles arrays of different shapes"""
    with pytest.raises(RuntimeError):
        lhs_shape = (2, 3)
        rhs_shape = (3, 2)
        dtypes = dtype.f16
        lhs = wrapper.randu(lhs_shape, dtypes)
        rhs = wrapper.randu(rhs_shape, dtypes)
        result = wrapper.mul(lhs, rhs)
        assert (
            wrapper.get_dims(result)[0 : len(lhs_shape)] == lhs_shape  # noqa
        ), f"Failed for shapes {lhs_shape} and {rhs_shape}"


def test_multiply_negative_shapes() -> None:
    """Test if multiplication handles arrays of negative shapes"""
    with pytest.raises(RuntimeError):
        lhs_shape = (2, -2)
        rhs_shape = (-2, 2)
        dtypes = dtype.f16
        lhs = wrapper.randu(lhs_shape, dtypes)
        rhs = wrapper.randu(rhs_shape, dtypes)
        result = wrapper.mul(lhs, rhs)
        assert (
            wrapper.get_dims(result)[0 : len(lhs_shape)] == lhs_shape  # noqa
        ), f"Failed for shapes {lhs_shape} and {rhs_shape}"


@pytest.mark.parametrize("dtype_name", util.get_all_types())
def test_multiply_supported_dtypes(dtype_name: dtype.Dtype) -> None:
    """Test multiplication operation across all supported data types."""
    util.check_type_supported(dtype_name)
    shape = (5, 5)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)
    result = wrapper.mul(lhs, rhs)
    assert dtype.c_api_value_to_dtype(wrapper.get_type(result)) == dtype_name, f"Failed for dtype: {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_multiply_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test multiplication operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        shape = (5, 5)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.mul(lhs, rhs)


def test_multiply_zero_sized_arrays() -> None:
    """Test multiplication with arrays where at least one array has zero size."""
    with pytest.raises(RuntimeError):
        zero_shape = (0, 5)
        normal_shape = (5, 5)
        zero_array = wrapper.randu(zero_shape, dtype.f32)
        normal_array = wrapper.randu(normal_shape, dtype.f32)

        result_rhs_zero = wrapper.mul(normal_array, zero_array)
        assert wrapper.get_dims(result_rhs_zero) == normal_shape

        result_lhs_zero = wrapper.mul(zero_array, normal_array)
        assert wrapper.get_dims(result_lhs_zero) == zero_shape


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((1, 5), (5, 1)),  # 2D with 2D inverse
        ((5, 5), (5, 1)),  # 2D with 2D
        ((5, 5), (1, 1)),  # 2D with 2D
        ((1, 1, 1), (5, 5, 5)),  # 3D with 3D
        ((5,), (5,)),  # 1D with 1D broadcast
    ],
)
def test_multiply_varying_dimensionality(shape_a: tuple, shape_b: tuple) -> None:
    """Test multiplication with arrays of varying dimensionality."""
    lhs = wrapper.randu(shape_a, dtype.f32)
    rhs = wrapper.randu(shape_b, dtype.f32)

    result = wrapper.mul(lhs, rhs)
    expected_shape = np.broadcast(np.empty(shape_a), np.empty(shape_b)).shape
    assert (
        wrapper.get_dims(result)[0 : len(expected_shape)] == expected_shape  # noqa
    ), f"Failed for shapes {shape_a} and {shape_b}"


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10),),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
def test_divide_shapes(shape: tuple) -> None:
    """Test division operation between two arrays of the same shape"""
    lhs = wrapper.randu(shape, dtype.f16)
    rhs = wrapper.randu(shape, dtype.f16)
    rhs = wrapper.add(rhs, wrapper.constant(0.001, shape, dtype.f16))

    result = wrapper.div(lhs, rhs)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa


def test_divide_different_shapes() -> None:
    """Test if division handles arrays of different shapes"""
    with pytest.raises(RuntimeError):
        lhs_shape = (2, 3)
        rhs_shape = (3, 2)
        dtypes = dtype.f16
        lhs = wrapper.randu(lhs_shape, dtypes)
        rhs = wrapper.randu(rhs_shape, dtypes)
        result = wrapper.div(lhs, rhs)
        expected_shape = np.broadcast(np.empty(lhs_shape), np.empty(rhs_shape)).shape
        assert (
            wrapper.get_dims(result)[0 : len(expected_shape)] == expected_shape  # noqa
        ), f"Failed for shapes {lhs_shape} and {rhs_shape}"


def test_divide_negative_shapes() -> None:
    """Test if division handles arrays of negative shapes"""
    with pytest.raises(RuntimeError):
        lhs_shape = (2, -2)
        rhs_shape = (-2, 2)
        dtypes = dtype.f16
        lhs = wrapper.randu(lhs_shape, dtypes)
        rhs = wrapper.randu(rhs_shape, dtypes)
        result = wrapper.div(lhs, rhs)
        expected_shape = np.broadcast(np.empty(lhs_shape), np.empty(rhs_shape)).shape
        assert (
            wrapper.get_dims(result)[0 : len(expected_shape)] == expected_shape  # noqa
        ), f"Failed for shapes {lhs_shape} and {rhs_shape}"


@pytest.mark.parametrize("dtype_name", util.get_all_types())
def test_divide_supported_dtypes(dtype_name: dtype.Dtype) -> None:
    """Test division operation across all supported data types."""
    util.check_type_supported(dtype_name)
    shape = (5, 5)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.add(rhs, wrapper.constant(0.001, shape, dtype_name))

    result = wrapper.div(lhs, rhs)
    assert dtype.c_api_value_to_dtype(wrapper.get_type(result)) == dtype_name, f"Failed for dtype: {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_divide_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test division operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        shape = (5, 5)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        # Prevent division by zero in unsupported dtype test
        rhs = wrapper.add(rhs, wrapper.constant(0.001, shape, invdtypes))

        wrapper.div(lhs, rhs)


def test_divide_zero_sized_arrays() -> None:
    """Test division with arrays where at least one array has zero size."""
    with pytest.raises(RuntimeError):
        zero_shape = (0, 5)
        normal_shape = (5, 5)
        zero_array = wrapper.randu(zero_shape, dtype.f32)
        normal_array = wrapper.randu(normal_shape, dtype.f32)

        result_rhs_zero = wrapper.div(normal_array, zero_array)
        assert wrapper.get_dims(result_rhs_zero) == normal_shape

        result_lhs_zero = wrapper.div(zero_array, normal_array)
        assert wrapper.get_dims(result_lhs_zero) == zero_shape


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((1, 5), (5, 1)),  # 2D with 2D inverse
        ((5, 5), (5, 1)),  # 2D with 2D
        ((5, 5), (1, 1)),  # 2D with 2D
        ((1, 1, 1), (5, 5, 5)),  # 3D with 3D
        ((5,), (5,)),  # 1D with 1D broadcast
    ],
)
def test_divide_varying_dimensionality(shape_a: tuple, shape_b: tuple) -> None:
    """Test division with arrays of varying dimensionality."""
    lhs = wrapper.randu(shape_a, dtype.f32)
    rhs = wrapper.randu(shape_b, dtype.f32)
    # Prevent division by zero for dimensional test
    rhs = wrapper.add(rhs, wrapper.constant(0.001, shape_b, dtype.f32))

    result = wrapper.div(lhs, rhs)
    expected_shape = np.broadcast(np.empty(shape_a), np.empty(shape_b)).shape
    assert (
        wrapper.get_dims(result)[0 : len(expected_shape)] == expected_shape  # noqa
    ), f"Failed for shapes {shape_a} and {shape_b}"
