import random

import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
from tests.utility_functions import check_type_supported, get_all_types, get_real_types


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
@pytest.mark.parametrize("dtype_name", get_all_types())
def test_and_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test and_ operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.and_(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_and_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test and_ operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)

        result = wrapper.and_(lhs, rhs)

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
@pytest.mark.parametrize("dtype_name", get_real_types())
def test_bitand_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test bitand operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    if dtype_name == dtype.f16 or dtype_name == dtype.f32:
        pytest.skip()
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.bitand(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_bitand_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test bitand operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.bitand(lhs, rhs)


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
@pytest.mark.parametrize("dtype_name", get_real_types())
def test_bitnot_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test bitnot operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    if dtype_name == dtype.f16 or dtype_name == dtype.f32:
        pytest.skip()
    out = wrapper.randu(shape, dtype_name)

    result = wrapper.bitnot(out)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_bitnot_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test bitnot operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        out = wrapper.randu(shape, invdtypes)
        wrapper.bitnot(out)


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
@pytest.mark.parametrize("dtype_name", get_real_types())
def test_bitor_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test bitor operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    if dtype_name == dtype.f16 or dtype_name == dtype.f32:
        pytest.skip()
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.bitor(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_bitor_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test bitor operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.bitor(lhs, rhs)


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
@pytest.mark.parametrize("dtype_name", get_real_types())
def test_bitxor_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test bitxor operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    if dtype_name == dtype.f16 or dtype_name == dtype.f32:
        pytest.skip()
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.bitxor(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_bitxor_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test bitxor operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.bitxor(lhs, rhs)


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
@pytest.mark.parametrize("dtype_name", get_all_types())
def test_eq_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test eq operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.eq(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_eq_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test eq operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.eq(lhs, rhs)


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
@pytest.mark.parametrize("dtype_name", get_all_types())
def test_ge_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test >= operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.ge(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_ge_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test >= operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.ge(lhs, rhs)


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
@pytest.mark.parametrize("dtype_name", get_all_types())
def test_gt_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test > operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.gt(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_gt_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test > operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.gt(lhs, rhs)


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
@pytest.mark.parametrize("dtype_name", get_all_types())
def test_le_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test <= operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.le(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_le_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test <= operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.le(lhs, rhs)


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
@pytest.mark.parametrize("dtype_name", get_all_types())
def test_lt_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test < operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.lt(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_lt_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test < operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.lt(lhs, rhs)


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
@pytest.mark.parametrize("dtype_name", get_all_types())
def test_neq_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test not equal operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.neq(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_neq_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test neq operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.neq(lhs, rhs)


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
@pytest.mark.parametrize("dtype_name", get_all_types())
def test_not_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test not operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    out = wrapper.randu(shape, dtype_name)

    result = wrapper.not_(out)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_not_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test not operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        out = wrapper.randu(shape, invdtypes)
        wrapper.not_(out)


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
@pytest.mark.parametrize("dtype_name", get_all_types())
def test_or_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test or operation between two arrays of the same shape"""
    check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.or_(lhs, rhs)

    assert (
        wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa
    ), f"failed for shape: {shape} and dtype {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_or_shapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test or operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.or_(lhs, rhs)
