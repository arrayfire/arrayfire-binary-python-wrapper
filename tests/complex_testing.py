import random

# import numpy as np
import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper

dtype_map = {
    # "int16": dtype.s16,
    # "int32": dtype.s32,
    # "int64": dtype.s64,
    # "uint8": dtype.u8,
    # "uint16": dtype.u16,
    # "uint32": dtype.u32,
    # "uint64": dtype.u64,
    # "float16": dtype.f16,
    "float32": dtype.f32,
    # 'float64': dtype.f64,
    # 'complex64': dtype.c64,
    # "complex32": dtype.c32,
    # "bool": dtype.b8,
    # "s16": dtype.s16,
    # "s32": dtype.s32,
    # "s64": dtype.s64,
    # "u8": dtype.u8,
    # "u16": dtype.u16,
    # "u32": dtype.u32,
    # "u64": dtype.u64,
    # "f16": dtype.f16,
    "f32": dtype.f32,
    # 'f64': dtype.f64,
    # "c32": dtype.c32,
    # 'c64': dtype.c64,
    # "b8": dtype.b8,
}


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
@pytest.mark.parametrize("dtype_name", dtype_map.values())
def test_complex_supported_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test complex operation across all supported data types."""
    tester = wrapper.randu(shape, dtype_name)
    result = wrapper.cplx(tester)
    assert wrapper.is_complex(result), f"Failed for dtype: {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_complex_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test complex operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        shape = (5, 5)
        out = wrapper.randu(shape, invdtypes)
        wrapper.cplx(out)


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
@pytest.mark.parametrize("dtype_name", dtype_map.values())
def test_complex2_supported_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test complex2 operation across all supported data types."""
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)
    result = wrapper.cplx2(lhs, rhs)
    assert wrapper.is_complex(result), f"Failed for dtype: {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_complex2_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test complex2 operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        shape = (5, 5)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)
        wrapper.cplx2(lhs, rhs)


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
def test_conj_supported_dtypes(shape: tuple) -> None:
    """Test conjugate operation for supported data types."""
    arr = wrapper.constant(7, shape, dtype.c32)
    result = wrapper.conjg(arr)
    assert wrapper.is_complex(result), f"Failed for shape: {shape}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_conj_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test conjugate operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        shape = (5, 5)
        arr = wrapper.randu(shape, invdtypes)
        wrapper.conjg(arr)


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
def test_imag_real_supported_dtypes(shape: tuple) -> None:
    """Test imaginary and real operations for supported data types."""
    arr = wrapper.randu(shape, dtype.c32)
    imaginary = wrapper.imag(arr)
    real = wrapper.real(arr)
    assert not wrapper.is_empty(imaginary), f"Failed for shape: {shape}"
    assert not wrapper.is_empty(real), f"Failed for shape: {shape}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_imag_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test conjugate operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        shape = (5, 5)
        arr = wrapper.randu(shape, invdtypes)
        wrapper.imag(arr)


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_real_unsupported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test real operation for unsupported data types."""
    with pytest.raises(RuntimeError):
        shape = (5, 5)
        arr = wrapper.randu(shape, invdtypes)
        wrapper.real(arr)
