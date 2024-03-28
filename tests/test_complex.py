import random

import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
from tests.utility_functions import check_type_supported, get_all_types, get_float_types, get_real_types


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
def test_complex_supported_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test complex operation across all supported data types."""
    check_type_supported(dtype_name)
    if dtype_name == dtype.f16:
        pytest.skip()
    tester = wrapper.randu(shape, dtype_name)
    result = wrapper.cplx(tester)
    assert wrapper.is_complex(result), f"Failed for dtype: {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.int32,
        dtype.complex32,
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
@pytest.mark.parametrize("dtype_name", get_real_types())
def test_complex2_supported_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test complex2 operation across all supported data types."""
    check_type_supported(dtype_name)
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)
    result = wrapper.cplx2(lhs, rhs)
    assert wrapper.is_complex(result), f"Failed for dtype: {dtype_name}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c32,
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
@pytest.mark.parametrize("dtypes", get_all_types())
def test_conj_supported_dtypes(shape: tuple, dtypes: dtype.Dtype) -> None:
    """Test conjugate operation for supported data types."""
    check_type_supported(dtypes)
    arr = wrapper.constant(7, shape, dtypes)
    result = wrapper.conjg(arr)
    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"Failed for shape: {shape}, and dtype: {dtypes}"  # noqa


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
@pytest.mark.parametrize("dtypes", get_all_types())
def test_imag_supported_dtypes(shape: tuple, dtypes: dtype.Dtype) -> None:
    """Test imaginary and real operations for supported data types."""
    check_type_supported(dtypes)
    arr = wrapper.randu(shape, dtypes)
    real = wrapper.real(arr)
    assert wrapper.is_real(real), f"Failed for shape: {shape}"


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
@pytest.mark.parametrize("dtypes", get_all_types())
def test_real_supported_dtypes(shape: tuple, dtypes: dtype.Dtype) -> None:
    """Test imaginary and real operations for supported data types."""
    check_type_supported(dtypes)
    arr = wrapper.randu(shape, dtypes)
    real = wrapper.real(arr)
    assert wrapper.is_real(real), f"Failed for shape: {shape}"
