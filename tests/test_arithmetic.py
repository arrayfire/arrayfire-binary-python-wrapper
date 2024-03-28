import random

import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
from tests.utility_functions import check_type_supported, get_all_types


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
def test_add_shapes(shape: tuple) -> None:
    """Test addition operation between two arrays of the same shape"""
    lhs = wrapper.randu(shape, dtype.f16)
    rhs = wrapper.randu(shape, dtype.f16)

    result = wrapper.add(lhs, rhs)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203, W291


def test_add_different_shapes() -> None:
    """Test if addition handles arrays of different shapes"""
    with pytest.raises(RuntimeError):
        lhs_shape = (2, 3)
        rhs_shape = (3, 2)
        dtypes = dtype.f16
        lhs = wrapper.randu(lhs_shape, dtypes)
        rhs = wrapper.randu(rhs_shape, dtypes)

        wrapper.add(lhs, rhs)


@pytest.mark.parametrize("dtype_name", get_all_types())
def test_add_supported_dtypes(dtype_name: dtype.Dtype) -> None:
    """Test addition operation across all supported data types."""
    check_type_supported(dtype_name)
    shape = (5, 5)  # Using a common shape for simplicity
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)
    result = wrapper.add(lhs, rhs)
    assert dtype.c_api_value_to_dtype(wrapper.get_type(result)) == dtype_name, f"Failed for dtype: {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_add_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test addition operation across all supported data types."""
    with pytest.raises(RuntimeError):
        shape = (5, 5)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        result = wrapper.add(lhs, rhs)
        assert dtype.c_api_value_to_dtype(wrapper.get_type(result)) == invdtypes, f"Didn't Fail for Dtype: {invdtypes}"


def test_add_zero_sized_arrays() -> None:
    """Test addition with arrays where at least one array has zero size."""
    with pytest.raises(RuntimeError):
        zero_shape = (0, 5)
        normal_shape = (5, 5)
        zero_array = wrapper.randu(zero_shape, dtype.f32)
        normal_array = wrapper.randu(normal_shape, dtype.f32)

        # Test addition when lhs is zero-sized
        result_lhs_zero = wrapper.add(zero_array, normal_array)
        assert wrapper.get_dims(result_lhs_zero) == zero_shape


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
def test_subtract_shapes(shape: tuple) -> None:
    """Test subtraction operation between two arrays of the same shape"""
    lhs = wrapper.randu(shape, dtype.f16)
    rhs = wrapper.randu(shape, dtype.f16)

    result = wrapper.sub(lhs, rhs)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203, W291


def test_subtract_different_shapes() -> None:
    """Test if subtraction handles arrays of different shapes"""
    with pytest.raises(RuntimeError):
        lhs_shape = (2, 3)
        rhs_shape = (3, 2)
        dtypes = dtype.f16
        lhs = wrapper.randu(lhs_shape, dtypes)
        rhs = wrapper.randu(rhs_shape, dtypes)

        wrapper.sub(lhs, rhs)


@pytest.mark.parametrize("dtype_name", get_all_types())
def test_subtract_supported_dtypes(dtype_name: dtype.Dtype) -> None:
    """Test subtraction operation across all supported data types."""
    check_type_supported(dtype_name)
    shape = (5, 5)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)
    result = wrapper.sub(lhs, rhs)
    assert dtype.c_api_value_to_dtype(wrapper.get_type(result)) == dtype_name, f"Failed for dtype: {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_subtract_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test subtraction operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        shape = (5, 5)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        result = wrapper.sub(lhs, rhs)
        assert result == invdtypes, f"Didn't Fail for Dtype: {invdtypes}"


def test_subtract_zero_sized_arrays() -> None:
    """Test subtraction with arrays where at least one array has zero size."""
    with pytest.raises(RuntimeError):
        zero_shape = (0, 5)
        normal_shape = (5, 5)
        zero_array = wrapper.randu(zero_shape, dtype.f32)
        normal_array = wrapper.randu(normal_shape, dtype.f32)

        result_lhs_zero = wrapper.sub(zero_array, normal_array)
        assert wrapper.get_dims(result_lhs_zero) == zero_shape
