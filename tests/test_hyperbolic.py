import random

import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper


dtype_map = {
    'int16': dtype.s16,
    'int32': dtype.s32,
    'int64': dtype.s64,
    'uint8': dtype.u8,
    'uint16': dtype.u16,
    'uint32': dtype.u32,
    'uint64': dtype.u64,
    'float16': dtype.f16,
    'float32': dtype.f32,
    "int16": dtype.s16,
    "int32": dtype.s32,
    "int64": dtype.s64,
    "uint8": dtype.u8,
    "uint16": dtype.u16,
    "uint32": dtype.u32,
    "uint64": dtype.u64,
    "float16": dtype.f16,
    "float32": dtype.f32,
    # 'float64': dtype.f64,
    # 'complex64': dtype.c64,
    # 'complex32': dtype.c32,
    'bool': dtype.b8,
    's16': dtype.s16,
    's32': dtype.s32,
    's64': dtype.s64,
    'u8': dtype.u8,
    'u16': dtype.u16,
    'u32': dtype.u32,
    'u64': dtype.u64,
    'f16': dtype.f16,
    'f32': dtype.f32,
    "bool": dtype.b8,
    "s16": dtype.s16,
    "s32": dtype.s32,
    "s64": dtype.s64,
    "u8": dtype.u8,
    "u16": dtype.u16,
    "u32": dtype.u32,
    "u64": dtype.u64,
    "f16": dtype.f16,
    "f32": dtype.f32,
    # 'f64': dtype.f64,
    # 'c32': dtype.c32,
    # 'c64': dtype.c64,
    'b8': dtype.b8,
    "b8": dtype.b8,
}


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10), ),
        (random.randint(1, 10),),
        (random.randint(1, 10),),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
@pytest.mark.parametrize("dtype_name", dtype_map.values())
def test_asinh_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test inverse hyperbolic sine operation across all supported data types."""
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.asinh(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_asinh_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test inverse hyperbolic sine operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        wrapper.asinh(wrapper.randu((10, 10), invdtypes))
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10), ),
        (random.randint(1, 10),),
        (random.randint(1, 10),),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
@pytest.mark.parametrize("dtype_name", dtype_map.values())
def test_acosh_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test inverse hyperbolic cosine operation across all supported data types."""
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.acosh(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_acosh_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test inverse hyperbolic cosine operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        wrapper.acosh(wrapper.randu((10, 10), invdtypes))
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10), ),
        (random.randint(1, 10),),
        (random.randint(1, 10),),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
@pytest.mark.parametrize("dtype_name", dtype_map.values())
def test_atanh_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test inverse hyperbolic tan operation across all supported data types."""
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.atanh(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_atanh_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test inverse hyperbolic tan operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        wrapper.atanh(wrapper.randu((10, 10), invdtypes))

@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10), ),
        (random.randint(1, 10),),
        (random.randint(1, 10),),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
@pytest.mark.parametrize("dtype_name", dtype_map.values())
def test_cosh_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test hyperbolic cosine operation across all supported data types."""
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.cosh(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_cosh_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test hyperbolic cosine operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        wrapper.cosh(wrapper.randu((10, 10), invdtypes))

@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10), ),
        (random.randint(1, 10),),
        (random.randint(1, 10),),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
@pytest.mark.parametrize("dtype_name", dtype_map.values())
def test_sinh_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test hyberbolic sin operation across all supported data types."""
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.sinh(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_sinh_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test hyperbolic sine operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        wrapper.sinh(wrapper.randu((10, 10), invdtypes))

@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10), ),
        (random.randint(1, 10),),
        (random.randint(1, 10),),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
@pytest.mark.parametrize("dtype_name", dtype_map.values())
def test_tanh_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test hyberbolic tan operation across all supported data types."""
    values = wrapper.randu(shape, dtype_name)
    result = wrapper.tanh(values)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape}"  # noqa


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_tanh_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test hyberbolic tan operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        wrapper.tanh(wrapper.randu((10, 10), invdtypes))
