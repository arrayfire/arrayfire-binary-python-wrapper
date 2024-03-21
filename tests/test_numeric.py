import random

import pytest
import numpy as np

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper

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
@pytest.mark.parametrize("dtype_name", util.get_all_types())
def test_abs_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test absolute value operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    out = wrapper.randu(shape, dtype_name)

    result = wrapper.abs_(out)
    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"

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
@pytest.mark.parametrize("dtype_name", util.get_all_types())
def test_abs_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test absolute value operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    out = wrapper.randu(shape, dtype_name)

    result = wrapper.abs_(out)
    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


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
@pytest.mark.parametrize("dtype_name", util.get_all_types())
def test_arg_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test arg operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    out = wrapper.randu(shape, dtype_name)

    result = wrapper.arg(out)
    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


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
@pytest.mark.parametrize("dtype_name", util.get_real_types())
def test_ceil_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test ceil operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    out = wrapper.randu(shape, dtype_name)

    result = wrapper.ceil(out)
    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"

@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c32,
    ],
)
def test_ceil_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test and_ operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        out = wrapper.randu(shape, invdtypes)

        result = wrapper.ceil(out)

        assert (
            wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
        ), f"failed for shape: {shape} and dtype {invdtypes}"
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
@pytest.mark.parametrize("dtype_name", util.get_all_types())
def test_maxof_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test maxof operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.maxof(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"
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
def test_maxof_varying_dimensionality(shape_a: tuple, shape_b: tuple) -> None:
    """Test maxof with arrays of varying dimensionality."""
    lhs = wrapper.randu(shape_a, dtype.f32)
    rhs = wrapper.randu(shape_b, dtype.f32)

    result = wrapper.maxof(lhs, rhs)
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
@pytest.mark.parametrize("dtype_name", util.get_all_types())
def test_minof_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test minof operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.minof(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"
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
def test_minof_varying_dimensionality(shape_a: tuple, shape_b: tuple) -> None:
    """Test minof with arrays of varying dimensionality."""
    lhs = wrapper.randu(shape_a, dtype.f32)
    rhs = wrapper.randu(shape_b, dtype.f32)

    result = wrapper.minof(lhs, rhs)
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
@pytest.mark.parametrize("dtype_name", util.get_real_types())
def test_mod_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test mod operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.mod(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"

@pytest.mark.parametrize("invdtypes", util.get_complex_types())
def test_mod_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test mod operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)

        result = wrapper.mod(lhs, rhs)

        assert (
            wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
        ), f"failed for shape: {shape} and dtype {invdtypes}"

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
@pytest.mark.parametrize("dtype_name", util.get_all_types())
def test_neg_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test arg operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    out = wrapper.randu(shape, dtype_name)

    result = wrapper.neg(out)
    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"

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
@pytest.mark.parametrize("dtype_name", util.get_real_types())
def test_rem_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test remainder operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.rem(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"
    
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
@pytest.mark.parametrize("dtype_name", util.get_real_types())
def test_round_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test round operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    out = wrapper.randu(shape, dtype_name)

    result = wrapper.round_(out)
    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"
    
@pytest.mark.parametrize("invdtypes", util.get_complex_types())
def test_round_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test round operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        out = wrapper.randu(shape, invdtypes)

        result = wrapper.round_(out)
        assert (
            wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
        ), f"failed for shape: {shape} and dtype {invdtypes}"
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
@pytest.mark.parametrize("dtype_name", util.get_real_types())
def test_sign_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test round operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    out = wrapper.randu(shape, dtype_name)

    result = wrapper.sign(out)
    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"

@pytest.mark.parametrize("invdtypes", util.get_complex_types())
def test_sign_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test sign operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        out = wrapper.randu(shape, invdtypes)

        result = wrapper.sign(out)
        assert (
            wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
        ), f"failed for shape: {shape} and dtype {invdtypes}"

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
@pytest.mark.parametrize("dtype_name", util.get_real_types())
def test_trunc_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test truncating operation for an array with varying shape"""
    util.check_type_supported(dtype_name)
    out = wrapper.randu(shape, dtype_name)

    result = wrapper.trunc(out)
    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"

@pytest.mark.parametrize("invdtypes", util.get_complex_types())
def test_trunc_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test trunc operation for an array with varrying shape and invalid dtypes"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        out = wrapper.randu(shape, invdtypes)

        result = wrapper.trunc(out)
        assert (
            wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
        ), f"failed for shape: {shape} and dtype {invdtypes}"
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
def test_hypot_shape_dtypes(shape: tuple) -> None:
    """Test hypotenuse operation between two arrays of the same shape"""
    lhs = wrapper.randu(shape, dtype.f32)
    rhs = wrapper.randu(shape, dtype.f32)

    result = wrapper.hypot(lhs, rhs, True)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype.f32}"
@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.int32,
        dtype.uint32,
    ],
)
def test_hypot_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test division operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        shape = (5, 5)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.hypot(rhs, lhs, True)
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
@pytest.mark.parametrize("dtype_name", util.get_real_types())
def test_clamp_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test clamp operation between two arrays of the same shape"""
    util.check_type_supported(dtype_name)
    og = wrapper.randu(shape, dtype_name)
    low = wrapper.randu(shape, dtype_name)
    high = wrapper.randu(shape, dtype_name)
    # talked to stefan about this, testing broadcasting is unnecessary
    result = wrapper.clamp(og, low, high, False)
    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"
