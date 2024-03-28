import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
from tests.utility_functions import check_type_supported, get_all_types, get_real_types


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (3,),
        (3, 3),
        (3, 3, 3),
        (3, 3, 3, 3),
    ],
)
@pytest.mark.parametrize("dtype_name", get_all_types())
def test_accum_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test accumulate operation across all supported data types."""
    check_type_supported(dtype_name)
    if dtype_name == dtype.f16:
        pytest.skip()
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.accum(values, 0)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
        2,
        3,
    ],
)
def test_accum_dims(dim: int) -> None:
    """Test accumulate dimensions operation"""
    shape = (3, 3)
    values = wrapper.randu(shape, dtype.f32)
    result = wrapper.accum(values, dim)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "invdim",
    [
        -1,
        5,
    ],
)
def test_accum_invdims(invdim: int) -> None:
    """Test accumulate invalid dimensions operation"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        values = wrapper.randu(shape, dtype.f32)
        result = wrapper.accum(values, invdim)
        assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (3,),
        (3, 3),
        (3, 3, 3),
        (3, 3, 3, 3),
    ],
)
@pytest.mark.parametrize("dtype_name", get_all_types())
def test_scan_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test scan operation across all supported data types."""
    check_type_supported(dtype_name)
    if dtype_name == dtype.f16:
        pytest.skip()
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.scan(values, 0, wrapper.BinaryOperator.ADD, True)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}, dtype {dtype_name}"  # noqa


@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
        2,
        3,
    ],
)
def test_scan_dims(dim: int) -> None:
    """Test scan dimensions operation"""
    shape = (3, 3)
    values = wrapper.randu(shape, dtype.f32)
    result = wrapper.scan(values, dim, wrapper.BinaryOperator.ADD, True)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for dimension: {dim}"  # noqa


@pytest.mark.parametrize(
    "invdim",
    [
        -1,
        5,
    ],
)
def test_scan_invdims(invdim: int) -> None:
    """Test scan invalid dimensions operation"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        values = wrapper.randu(shape, dtype.f32)
        result = wrapper.scan(values, invdim, wrapper.BinaryOperator.ADD, True)
        assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "binaryOp",
    [
        0,
        1,
        2,
        3,
    ],
)
def test_scan_binaryOp(binaryOp: int) -> None:
    """Test scan dimensions operation"""
    shape = (3, 3)
    values = wrapper.randu(shape, dtype.f32)
    result = wrapper.scan(values, 0, wrapper.BinaryOperator(binaryOp), True)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for operation: {binaryOp}"  # noqa


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (3,),
        (3, 3),
        (3, 3, 3),
        (3, 3, 3, 3),
    ],
)
@pytest.mark.parametrize("dtype_name", get_real_types())
def test_scan_by_key_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test scan_by_key operation across all supported data types."""
    check_type_supported(dtype_name)
    if (
        dtype_name == dtype.f16
        or dtype_name == dtype.f32
        or dtype_name == dtype.uint16
        or dtype_name == dtype.uint8
        or dtype_name == dtype.int16
    ):
        pytest.skip()
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.scan_by_key(values, values, 0, wrapper.BinaryOperator.ADD, True)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}, dtype {dtype_name}"  # noqa


@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
        2,
        3,
    ],
)
def test_scan_by_key_dims(dim: int) -> None:
    """Test scan_by_key dimensions operation"""
    shape = (3, 3)
    values = wrapper.randu(shape, dtype.int32)
    result = wrapper.scan_by_key(values, values, dim, wrapper.BinaryOperator.ADD, True)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for dimension: {dim}"  # noqa


@pytest.mark.parametrize(
    "invdim",
    [
        -1,
        5,
    ],
)
def test_scan_by_key_invdims(invdim: int) -> None:
    """Test scan_by_key invalid dimensions operation"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        values = wrapper.randu(shape, dtype.int32)
        result = wrapper.scan_by_key(values, values, invdim, wrapper.BinaryOperator.ADD, True)
        assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "binaryOp",
    [
        0,
        1,
        2,
        3,
    ],
)
def test_scan_by_key_binaryOp(binaryOp: int) -> None:
    """Test scan_by_key dimensions operation"""
    shape = (3, 3)
    values = wrapper.randu(shape, dtype.int32)
    result = wrapper.scan_by_key(values, values, 0, wrapper.BinaryOperator(binaryOp), True)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for operation: {binaryOp}"  # noqa


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (3,),
        (3, 3),
        (3, 3, 3),
        (3, 3, 3, 3),
    ],
)
@pytest.mark.parametrize("dtype_name", get_real_types())
def test_where_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test where operation across all supported data types."""
    check_type_supported(dtype_name)
    if dtype_name == dtype.f16:
        pytest.skip()
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.where(values)
    assert wrapper.get_dims(result)[0] == 3 ** len(shape), f"failed for shape: {shape}, dtype {dtype_name}"  # noqa
