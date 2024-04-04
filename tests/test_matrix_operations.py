import random

import pytest

import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.dtypes import Dtype, b8, c32, c64, f16, f32, f64, s16, s32, s64, u8, u16, u32, u64

from .utility_functions import check_type_supported


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (10, 10), (100, 100), (1000, 1000), (10000, 10000)],
)
def test_det_type(shape: tuple) -> None:
    """Test if det returns a complex number"""
    arr = wrapper.randn(shape, f32)
    determinant = wrapper.det(arr)

    assert isinstance(determinant, complex)


@pytest.mark.parametrize(
    "shape",
    [
        (15, 17),
        (105, 325),
        (567, 803),
        (5324, 7865),
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (10000, 10000, 10000),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
        (10000, 10000, 10000, 10000),
    ],
)
def test_det_invalid_shape(shape: tuple) -> None:
    """Test if det can properly handle invalid shapes"""
    with pytest.raises(RuntimeError):
        arr = wrapper.randn(shape, f32)
        wrapper.det(arr)


@pytest.mark.parametrize("dtype", [s16, s32, s64, u8, u16, u32, u64, f16, b8])
def test_det_invalid_dtype(dtype: Dtype) -> None:
    """Test if det can properly handle invalid dtypes"""
    with pytest.raises(RuntimeError):
        shape = (10, 10)

        arr = wrapper.identity(shape, dtype)
        wrapper.det(arr)


@pytest.mark.parametrize("dtype", [f32, f64, c32, c64])
def test_det_valid_dtype(dtype: Dtype) -> None:
    """Test if det can properly handle all valid dtypes"""
    check_type_supported(dtype)
    shape = (10, 10)

    arr = wrapper.identity(shape, dtype)
    determinant = wrapper.det(arr)

    assert isinstance(determinant, complex)


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (10, 10), (100, 100), (1000, 1000), (10000, 10000)],
)
def test_inverse_type(shape: tuple) -> None:
    """Test if inverse returns an AFArray"""
    arr = wrapper.randn(shape, f32)
    inv = wrapper.inverse(arr, wrapper.MatProp(0))

    assert isinstance(inv, AFArray)


@pytest.mark.parametrize(
    "shape",
    [
        (15, 17),
        (105, 325),
        (567, 803),
        (5324, 7865),
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (10000, 10000, 10000),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
        (10000, 10000, 10000, 10000),
    ],
)
def test_inverse_invalid_shape(shape: tuple) -> None:
    """Test if inverse can properly handle invalid shapes"""
    with pytest.raises(RuntimeError):
        arr = wrapper.randn(shape, f32)
        wrapper.inverse(arr, wrapper.MatProp(0))


@pytest.mark.parametrize("dtype", [s16, s32, s64, u8, u16, u32, u64, f16, b8])
def test_inverse_invalid_dtype(dtype: Dtype) -> None:
    """Test if inverse can properly handle invalid dtypes"""
    with pytest.raises(RuntimeError):
        shape = (10, 10)

        arr = wrapper.identity(shape, dtype)
        wrapper.inverse(arr, wrapper.MatProp(0))


@pytest.mark.parametrize("dtype", [f32, f64, c32, c64])
def test_inverse_valid_dtype(dtype: Dtype) -> None:
    """Test if inverse can properly handle all valid dtypes"""
    check_type_supported(dtype)
    shape = (10, 10)

    arr = wrapper.identity(shape, dtype)
    wrapper.inverse(arr, wrapper.MatProp(0))


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (10, 10), (100, 100), (1000, 1000), (10000, 10000)],
)
def test_norm_output_type(shape: tuple) -> None:
    """Test if norm returns a float"""
    arr = wrapper.randn(shape, f32)
    nor = wrapper.norm(arr, wrapper.Norm(2), 1, 1)

    assert isinstance(nor, float)


@pytest.mark.parametrize(
    "norm",
    [0, 1, 2, 3, 4, 5, 7],  # VECTOR_1  # VECTOR_INF  # VECTOR_2  # VECTOR_3  # MATRIX_1  # MATRIX_INF  # MATRIX_L_PQ
)
def test_norm_types(norm: wrapper.Norm) -> None:
    """Test if norm can handle all valid norm types"""
    shape = (3, 1)
    arr = wrapper.randn(shape, f32)
    nor = wrapper.norm(arr, wrapper.Norm(norm), 1, 2)

    assert isinstance(nor, float)


@pytest.mark.parametrize(
    "shape",
    [
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (10000, 10000, 10000),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
        (10000, 10000, 10000, 10000),
    ],
)
def test_norm_invalid_shape(shape: tuple) -> None:
    """Test if norm can properly handle invalid shapes"""
    with pytest.raises(RuntimeError):
        arr = wrapper.randn(shape, f32)
        wrapper.norm(arr, wrapper.Norm(0), 1, 1)


@pytest.mark.parametrize("dtype", [s16, s32, s64, u8, u16, u32, u64, f16, b8])
def test_norm_invalid_dtype(dtype: Dtype) -> None:
    """Test if norm can properly handle invalid dtypes"""
    with pytest.raises(RuntimeError):
        shape = (10, 10)

        arr = wrapper.identity(shape, dtype)
        wrapper.norm(arr, wrapper.Norm(0), 1, 1)


@pytest.mark.parametrize("dtype", [f32, f64, c32, c64])
def test_norm_valid_dtype(dtype: Dtype) -> None:
    """Test if norm can properly handle all valid dtypes"""
    check_type_supported(dtype)
    shape = (10, 10)

    arr = wrapper.identity(shape, dtype)
    wrapper.norm(arr, wrapper.Norm(0), 1, 1)


# pinverse tests
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (10, 10),
        (100, 100),
        (1000, 1000),
        (1, 1, 1),
        (10, 10, 10),
        (100, 100, 100),
        (1, 1, 1, 1),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
    ],
)
def test_pinverse_output_type(shape: tuple) -> None:
    """Test if pinverse returns an AFArray"""
    arr = wrapper.randn(shape, f32)
    pin = wrapper.pinverse(arr, 1e-6, wrapper.MatProp(0))

    assert isinstance(pin, AFArray)


@pytest.mark.parametrize("dtype", [s16, s32, s64, u8, u16, u32, u64, f16, b8])
def test_pinverse_invalid_dtype(dtype: Dtype) -> None:
    """Test if pinverse can properly handle invalid dtypes"""
    with pytest.raises(RuntimeError):
        shape = (10, 10)

        arr = wrapper.identity(shape, dtype)
        wrapper.pinverse(arr, 1e-6, wrapper.MatProp(0))


@pytest.mark.parametrize("dtype", [f32, f64, c32, c64])
def test_pinverse_valid_dtype(dtype: Dtype) -> None:
    """Test if pinverse can properly handle all valid dtypes"""
    check_type_supported(dtype)
    shape = (10, 10)

    arr = wrapper.identity(shape, dtype)
    pin = wrapper.pinverse(arr, 1e-6, wrapper.MatProp(0))

    assert isinstance(pin, AFArray)


@pytest.mark.parametrize("tolerance", [-0.0001, -1, -10, -100, -1000])
def test_pinverse_invalid_tol(tolerance: int) -> None:
    """Test if pinverse can properly handle invalid tolerance values"""
    with pytest.raises(RuntimeError):
        shape = (10, 10)

        arr = wrapper.identity(shape, f32)
        wrapper.pinverse(arr, tolerance, wrapper.MatProp(0))


# rank tests
@pytest.mark.parametrize(
    "shape",
    [(1, 1), (10, 10), (100, 100), (1000, 1000), (random.randint(1, 1000), random.randint(1, 1000))],
)
def test_rank_output_type(shape: tuple) -> None:
    """Test if rank returns an AFArray"""
    arr = wrapper.randn(shape, f32)
    rk = wrapper.rank(arr, 1e-6)

    assert isinstance(rk, int)


@pytest.mark.parametrize(
    "shape",
    [
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (10000, 10000, 10000),
        (random.randint(1, 1000), random.randint(1, 1000), random.randint(1, 1000)),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
        (10000, 10000, 10000, 10000),
        (random.randint(1, 1000), random.randint(1, 1000), random.randint(1, 1000), random.randint(1, 10000)),
    ],
)
def test_rank_invalid_shape(shape: tuple) -> None:
    """Test if rank can properly handle invalid shapes"""
    with pytest.raises(RuntimeError):
        arr = wrapper.randn(shape, f32)
        wrapper.rank(arr, 1e-6)


@pytest.mark.parametrize("dtype", [s16, s32, s64, u8, u16, u32, u64, f16, b8])
def test_rank_invalid_dtype(dtype: Dtype) -> None:
    """Test if rank can properly handle invalid dtypes"""
    with pytest.raises(RuntimeError):
        shape = (10, 10)

        arr = wrapper.identity(shape, dtype)
        wrapper.rank(arr, 1e-6)


@pytest.mark.parametrize("dtype", [f32, f64, c32, c64])
def test_rank_valid_dtype(dtype: Dtype) -> None:
    """Test if rank can properly handle all valid dtypes"""
    check_type_supported(dtype)
    shape = (10, 10)

    arr = wrapper.identity(shape, dtype)
    rk = wrapper.rank(arr, 1e-6)

    assert isinstance(rk, int)
