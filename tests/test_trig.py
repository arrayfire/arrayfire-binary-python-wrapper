import random

import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
from tests.utility_functions import check_type_supported, get_all_types, get_float_types


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
def test_asin_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test inverse sine operation across all supported data types."""
    check_type_supported(dtype_name)
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.asin(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


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
def test_acos_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test inverse cosine operation across all supported data types."""
    check_type_supported(dtype_name)
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.acos(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


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
def test_atan_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test inverse tan operation across all supported data types."""
    check_type_supported(dtype_name)
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.atan(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


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
@pytest.mark.parametrize("dtype_name", get_float_types())
def test_atan2_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test inverse tan operation across all supported data types."""
    check_type_supported(dtype_name)
    if dtype_name == dtype.f16:
        pytest.skip()
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)
    result = wrapper.atan2(lhs, rhs)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.int16,
        dtype.bool,
    ],
)
def test_atan2_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test inverse tan operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        wrapper.atan2(wrapper.randu((10, 10), invdtypes), wrapper.randu((10, 10), invdtypes))


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
def test_cos_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test cosine operation across all supported data types."""
    check_type_supported(dtype_name)
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.cos(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


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
def test_sin_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test sin operation across all supported data types."""
    check_type_supported(dtype_name)
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.sin(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


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
def test_tan_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test tan operation across all supported data types."""
    check_type_supported(dtype_name)
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.tan(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa
