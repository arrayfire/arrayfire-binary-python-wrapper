import pytest

import arrayfire_wrapper.dtypes as dtypes
import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.dtypes import Dtype, c32, c64, f16, f32, f64, s16, s32, s64, u8, u16, u32, u64

from .utility_functions import check_type_supported


@pytest.mark.parametrize(
    "shape",
    [(10, 10), (100, 100), (1000, 1000)],
)
def test_lu_square_shape(shape: tuple) -> None:
    """Test if the lu function returns values with the correct shape if the input matrix is a square matrix"""
    dtype = dtypes.f32

    arr = wrapper.randu(shape, dtype)
    L, U, P = wrapper.lu(arr)
    assert wrapper.get_dims(L)[: len(shape)] == shape
    assert wrapper.get_dims(U)[: len(shape)] == shape
    assert wrapper.get_dims(P) == (shape[0], 1, 1, 1)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (10, 19),
        (100, 101),
        (1000, 1960),
    ],
)
def test_lu_non_square_shape_1(shape: tuple) -> None:
    """Test if the lu function returns matrices with the correct shape if rectangular matrix"""
    dtype = dtypes.f32
    arr = wrapper.randu(shape, dtype)
    L, U, P = wrapper.lu(arr)

    assert wrapper.get_dims(L)[: len(shape)] == (shape[0], shape[0])
    assert wrapper.get_dims(U)[: len(shape)] == shape
    assert wrapper.get_dims(P) == (shape[0], 1, 1, 1)


@pytest.mark.parametrize(
    "shape",
    [(3, 2), (19, 10), (101, 100), (1960, 1000)],
)
def test_lu_non_square_shape_2(shape: tuple) -> None:
    """Test if the lu function returns matrices with the correct shape if rectangular matrix"""
    dtype = dtypes.f32
    arr = wrapper.randu(shape, dtype)
    L, U, P = wrapper.lu(arr)

    assert wrapper.get_dims(L)[: len(shape)] == shape
    assert wrapper.get_dims(U)[: len(shape)] == (shape[1], shape[1])
    assert wrapper.get_dims(P) == (shape[0], 1, 1, 1)


@pytest.mark.parametrize(
    "shape",
    [
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
    ],
)
def test_lu_invalid_shape(shape: tuple) -> None:
    """Test if the lu function properly handles an invalid shape given"""
    with pytest.raises(RuntimeError):

        dtype = dtypes.f32
        arr = wrapper.randu(shape, dtype)
        wrapper.lu(arr)


@pytest.mark.parametrize(
    "dtype",
    [f32, f64, c32, c64],
)
def test_lu_valid_dtype(dtype: Dtype) -> None:
    """Test if the lu function runs properly with the correct dtypes"""
    check_type_supported(dtype)

    shape = (10, 10)

    arr = wrapper.randu(shape, dtype)
    L, U, P = wrapper.lu(arr)

    assert dtypes.c_api_value_to_dtype(wrapper.get_type(L)) == dtype
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(U)) == dtype
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(P)) == dtypes.int32


@pytest.mark.parametrize("dtype", [f16, s16, s32, s64, u8, u16, u32, u64])
def test_lu_invalid_dtype(dtype: Dtype) -> None:
    """Test if the lu function runs properly with invalid dtypes"""
    check_type_supported(dtype)

    with pytest.raises(RuntimeError):
        shape = (10, 10)

        arr = wrapper.randu(shape, dtype)
        wrapper.lu(arr)


@pytest.mark.parametrize(
    "shape",
    [(10, 10), (100, 100), (1000, 1000)],
)
def test_lu_inplace_square_shape(shape: tuple) -> None:
    """Test if the lu_inplace function returns a pivot matrix with the correct shape if the input is mxm"""
    dtype = dtypes.f32

    arr = wrapper.randu(shape, dtype)
    P = wrapper.lu_inplace(arr, True)
    assert wrapper.get_dims(P) == (shape[0], 1, 1, 1)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (10, 19),
        (100, 101),
        (1000, 1960),
    ],
)
def test_lu_inplace_non_square_shape_1(shape: tuple) -> None:
    """Test lu_inplace for correct pivot matrix shape with rectangular input, shape[0] < shape[1]"""
    dtype = dtypes.f32
    arr = wrapper.randu(shape, dtype)
    P = wrapper.lu_inplace(arr, True)

    assert wrapper.get_dims(P) == (shape[0], 1, 1, 1)


@pytest.mark.parametrize(
    "shape",
    [(3, 2), (19, 10), (101, 100), (1960, 1000)],
)
def test_lu_inplace_non_square_shape_2(shape: tuple) -> None:
    """Test lu_inplace for correct pivot shape with input where shape[0] > shape[1]"""
    dtype = dtypes.f32
    arr = wrapper.randu(shape, dtype)
    P = wrapper.lu_inplace(arr, True)

    assert wrapper.get_dims(P) == (shape[1], 1, 1, 1)


@pytest.mark.parametrize(
    "shape",
    [
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
    ],
)
def test_lu_inplace_invalid_shape(shape: tuple) -> None:
    """Test if the lu_inplace function properly handles an invalid shape"""
    with pytest.raises(RuntimeError):

        dtype = dtypes.f32
        arr = wrapper.randu(shape, dtype)
        wrapper.lu_inplace(arr, True)


@pytest.mark.parametrize(
    "dtype",
    [f32, f64, c32, c64],
)
def test_lu_inplace_valid_dtype(dtype: Dtype) -> None:
    """Tests if the lu_inplace function returns a pivot matrix with the correct dtype"""
    check_type_supported(dtype)

    shape = (10, 10)

    arr = wrapper.randu(shape, dtype)
    P = wrapper.lu_inplace(arr, True)
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(P)) == dtypes.int32


@pytest.mark.parametrize(
    "dtype",
    [f16, s16, s32, s64, u8, u16, u32, u64],
)
def test_lu_inplace_invalid_dtype(dtype: Dtype) -> None:
    """Test if the lu_inplace function properly handles an invalid dtype"""
    check_type_supported(dtype)

    with pytest.raises(RuntimeError):
        shape = (10, 10)

        arr = wrapper.randu(shape, dtype)
        wrapper.lu_inplace(arr, True)


@pytest.mark.parametrize(
    "shape",
    [(10, 10), (100, 100), (1000, 1000)],
)
def test_qr_square_shape(shape: tuple) -> None:
    """Test qr function for correct q, r, tau shapes with square input matrix"""
    dtype = dtypes.f32

    arr = wrapper.randu(shape, dtype)
    Q, R, tau = wrapper.qr(arr)
    assert wrapper.get_dims(Q)[: len(shape)] == shape
    assert wrapper.get_dims(R)[: len(shape)] == shape
    assert wrapper.get_dims(tau) == (shape[0], 1, 1, 1)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (10, 19),
        (100, 101),
        (1000, 1960),
    ],
)
def test_qr_non_square_shape_1(shape: tuple) -> None:
    """Test qr function for correct shapes of q, r, tau with rectangular input, shape[0] < shape[1]"""
    dtype = dtypes.f32
    arr = wrapper.randu(shape, dtype)
    Q, R, tau = wrapper.qr(arr)

    assert wrapper.get_dims(Q)[: len(shape)] == (shape[0], shape[0])
    assert wrapper.get_dims(R)[: len(shape)] == shape
    assert wrapper.get_dims(tau) == (min(shape[0], shape[1]), 1, 1, 1)


@pytest.mark.parametrize(
    "shape",
    [(3, 2), (19, 10), (101, 100), (1960, 1000)],
)
def test_qr_non_square_shape_2(shape: tuple) -> None:
    """Test qr for correct q, r, tau shapes with rectangular input, shape[0] > shape[1]"""
    dtype = dtypes.f32
    arr = wrapper.randu(shape, dtype)
    Q, R, tau = wrapper.qr(arr)

    assert wrapper.get_dims(Q)[: len(shape)] == (shape[0], shape[0])
    assert wrapper.get_dims(R)[: len(shape)] == shape
    assert wrapper.get_dims(tau) == (min(shape[0], shape[1]), 1, 1, 1)


@pytest.mark.parametrize(
    "shape",
    [
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
    ],
)
def test_qr_invalid_shape(shape: tuple) -> None:
    """Test if the qr function properly handles a matrix with an invalid shape"""

    with pytest.raises(RuntimeError):

        dtype = dtypes.f32
        arr = wrapper.randu(shape, dtype)
        wrapper.qr(arr)


@pytest.mark.parametrize(
    "dtype",
    [f32, f64, c32, c64],
)
def test_qr_valid_dtype(dtype: Dtype) -> None:
    """Test if the qr function runs properly with the correct dtypes"""
    check_type_supported(dtype)

    shape = (10, 10)

    arr = wrapper.randu(shape, dtype)
    Q, R, tau = wrapper.qr(arr)

    assert dtypes.c_api_value_to_dtype(wrapper.get_type(Q)) == dtype
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(R)) == dtype
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(tau)) == dtype


@pytest.mark.parametrize(
    "dtype",
    [f16, s16, s32, s64, u8, u16, u32, u64],
)
def test_qr_invalid_dtype(dtype: Dtype) -> None:
    """Test if the qr function properly handles invalid dtypes"""
    check_type_supported(dtype)

    with pytest.raises(RuntimeError):
        shape = (10, 10)

        arr = wrapper.randu(shape, dtype)
        wrapper.qr(arr)


@pytest.mark.parametrize(
    "shape",
    [(10, 10), (100, 100), (1000, 1000)],
)
def test_svd_square_shape(shape: tuple) -> None:
    """Test svd for correct s, vt, p shapes with square input matrix"""
    dtype = dtypes.f32

    arr = wrapper.randu(shape, dtype)
    u, s, vt = wrapper.svd(arr)
    assert wrapper.get_dims(u)[: len(shape)] == shape
    assert wrapper.get_dims(vt)[: len(shape)] == shape
    assert wrapper.get_dims(s) == (shape[0], 1, 1, 1)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (10, 19),
        (100, 101),
        (1000, 1960),
    ],
)
def test_svd_non_square_shape_1(shape: tuple) -> None:
    """Test svd for correct u, s, vt shapes with rectangular input, shape[0] < shape[1]"""
    dtype = dtypes.f32
    arr = wrapper.randu(shape, dtype)
    u, s, vt = wrapper.svd(arr)

    assert wrapper.get_dims(u)[: len(shape)] == (shape[0], shape[0])
    assert wrapper.get_dims(s) == (shape[0], 1, 1, 1)
    assert wrapper.get_dims(vt)[: len(shape)] == (shape[1], shape[1])


@pytest.mark.parametrize(
    "shape",
    [(3, 2), (19, 10), (101, 100), (1960, 1000)],
)
def test_svd_non_square_shape_2(shape: tuple) -> None:
    """Test svd for correct u, s, vt shapes with input where shape[1] > shape[0]"""
    dtype = dtypes.f32
    arr = wrapper.randu(shape, dtype)
    u, s, vt = wrapper.svd(arr)

    assert wrapper.get_dims(u)[: len(shape)] == (shape[0], shape[0])
    assert wrapper.get_dims(s) == (shape[1], 1, 1, 1)
    assert wrapper.get_dims(vt)[: len(shape)] == (shape[1], shape[1])


@pytest.mark.parametrize(
    "shape",
    [
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
    ],
)
def test_svd_invalid_shape(shape: tuple) -> None:
    """Test if the svd function properly handles an invalid shape"""
    with pytest.raises(RuntimeError):

        dtype = dtypes.f32
        arr = wrapper.randu(shape, dtype)
        wrapper.svd(arr)


@pytest.mark.parametrize(
    "dtype",
    [f32, f64, c32, c64],
)
def test_svd_valid_dtype(dtype: Dtype) -> None:
    """Test if the svd function runs properly with the correct dtypes"""
    check_type_supported(dtype)

    shape = (10, 10)

    arr = wrapper.randu(shape, dtype)
    u, s, vt = wrapper.svd(arr)

    valid_outputs = [dtypes.complex32, dtypes.float32, dtypes.complex64, dtypes.float64]
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(u)) in valid_outputs
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(s)) in valid_outputs
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(vt)) in valid_outputs


@pytest.mark.parametrize(
    "dtype",
    [f16, s16, s32, s64, u8, u16, u32, u64],
)
def test_svd_invalid_dtype(dtype: Dtype) -> None:
    """Test if the svd function properly handles invalid dtypes"""
    check_type_supported(dtype)

    with pytest.raises(RuntimeError):
        shape = (10, 10)

        arr = wrapper.randu(shape, dtype)
        wrapper.svd(arr)


@pytest.mark.parametrize(
    "shape",
    [(10, 10), (100, 100), (1000, 1000)],
)
def test_svd_inplace_square_shape(shape: tuple) -> None:
    """Test if the svd_inplace function returns values with the correct shape if input matrix is square matrix"""
    dtype = dtypes.f32

    arr = wrapper.randu(shape, dtype)
    u, s, vt, arr = wrapper.svd_inplace(arr)
    assert wrapper.get_dims(u)[: len(shape)] == shape
    assert wrapper.get_dims(vt)[: len(shape)] == shape
    assert wrapper.get_dims(s) == (shape[0], 1, 1, 1)
    assert wrapper.get_dims(arr)[: len(shape)] == shape


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (10, 19),
        (100, 101),
        (1000, 1960),
    ],
)
def test_svd_inplace_non_square_shape_1(shape: tuple) -> None:
    """Test svd_inplace for correct u, s, vt, arr shapes with rectangular input, shape[0] < shape[1]"""
    dtype = dtypes.f32
    arr = wrapper.randu(shape, dtype)
    u, s, vt, arr = wrapper.svd_inplace(arr)

    assert wrapper.get_dims(u)[: len(shape)] == (shape[0], shape[0])
    assert wrapper.get_dims(s) == (shape[0], 1, 1, 1)
    assert wrapper.get_dims(vt)[: len(shape)] == (shape[1], shape[1])
    assert wrapper.get_dims(arr)[: len(shape)] == shape


@pytest.mark.parametrize(
    "shape",
    [(3, 2), (19, 10), (101, 100), (1960, 1000)],
)
def test_svd_inplace_non_square_shape_2(shape: tuple) -> None:
    """Test svd_inplace for correct u, s, vt, arr shapes with rectangular input, shape[1] < shape[0]"""
    dtype = dtypes.f32
    arr = wrapper.randu(shape, dtype)
    u, s, vt, arr = wrapper.svd_inplace(arr)

    assert wrapper.get_dims(u)[: len(shape)] == (shape[0], shape[0])
    assert wrapper.get_dims(s) == (shape[1], 1, 1, 1)
    assert wrapper.get_dims(vt)[: len(shape)] == (shape[1], shape[1])
    assert wrapper.get_dims(arr)[: len(shape)] == shape


@pytest.mark.parametrize(
    "shape",
    [
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
    ],
)
def test_svd_inplace_invalid_shape(shape: tuple) -> None:
    """Test if the svd_inplace function properly handles an invalid shape"""
    with pytest.raises(RuntimeError):

        dtype = dtypes.f32
        arr = wrapper.randu(shape, dtype)
        wrapper.svd_inplace(arr)


@pytest.mark.parametrize(
    "dtype",
    [f32, f64, c32, c64],
)
def test_svd_inplace_valid_dtype(dtype: Dtype) -> None:
    """Tests if the svd_inplace function returns a pivot matrix with the correct dtype"""
    check_type_supported(dtype)

    shape = (10, 10)

    arr = wrapper.randu(shape, dtype)
    u, s, vt, arr = wrapper.svd_inplace(arr)

    valid_outputs = [dtypes.complex32, dtypes.float32, dtypes.complex64, dtypes.float64]
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(u)) in valid_outputs
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(s)) in valid_outputs
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(vt)) in valid_outputs
    assert dtypes.c_api_value_to_dtype(wrapper.get_type(arr)) == dtype


@pytest.mark.parametrize(
    "dtype",
    [f16, s16, s32, s64, u8, u16, u32, u64],
)
def test_svd_inplace_invalid_dtype(dtype: Dtype) -> None:
    """Tests if the svd_inplace properly handles invalid dtypes"""
    check_type_supported(dtype)

    with pytest.raises(RuntimeError):
        shape = (10, 10)

        arr = wrapper.randu(shape, dtype)
        wrapper.svd_inplace(arr)
