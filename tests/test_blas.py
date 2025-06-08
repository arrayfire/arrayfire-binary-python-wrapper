import random

import pytest

import arrayfire_wrapper.dtypes as dtypes
import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._constants import MatProp


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (100,),
        (1000,),
        (10000,),
        (1, 1),
        (10, 10),
        (100, 100),
        (1000, 1000),
        (1, 1, 1),
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (1, 1, 1, 1),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
    ],
)
def test_dot_res(shape: tuple) -> None:
    """Test if the dot product outputs an AFArray with a dimension of 1"""
    dtype = dtypes.f32
    shape = (10,)
    x = wrapper.randu(shape, dtype)

    result = wrapper.dot(x, x, MatProp.NONE, MatProp.NONE)

    assert isinstance(result, AFArray)


@pytest.mark.parametrize(
    "shape_pairs",
    [
        [(5, 1), (1, 6)],
        [(10, 10), (9, 10)],
        [(9, 8), (10, 10)],
        [(random.randint(1, 10), 100), (1000, random.randint(1, 10))],
        [(random.randint(1, 10), 100000), (2, random.randint(1, 10))],
    ],
)
def test_dot_invalid_shape_comp(shape_pairs: list) -> None:
    """Test if an improper shape pair is properly handled"""
    with pytest.raises(RuntimeError):
        dtype = dtypes.f32
        x = wrapper.randu(shape_pairs[0], dtype)
        y = wrapper.randu(shape_pairs[1], dtype)

        wrapper.dot(x, y, MatProp.NONE, MatProp.NONE)


def test_dot_empty_vector() -> None:
    """Test if an empty array passed into the dot product is properly handled"""
    with pytest.raises(RuntimeError):
        empty_shape = (0,)
        dtype = dtypes.f32

        x = wrapper.randu(empty_shape, dtype)
        wrapper.dot(x, x, MatProp.NONE, MatProp.NONE)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (100,),
        (1000,),
        (10000,),
        (1, 1),
        (10, 10),
        (100, 100),
        (1000, 1000),
        (1, 1, 1),
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (1, 1, 1, 1),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
    ],
)
def test_dot_diff_dtype(shape: tuple) -> None:
    """Test of dot product of arrays of different dtypes is properly handled"""
    with pytest.raises(RuntimeError):
        x = wrapper.randu(shape, dtypes.f32)
        y = wrapper.randu(shape, dtypes.c32)

        wrapper.dot(x, y, MatProp.NONE, MatProp.NONE)


@pytest.mark.parametrize(
    "shape",
    [(1,), (10,), (100,), (1000,)],
)
@pytest.mark.parametrize(
    "dtype_index",
    [i for i in range(13)],
)
def test_dot_invalid_dtype(shape: tuple, dtype_index: int) -> None:
    """Test if improper dtypes are properly handled"""
    if dtype_index in [12, 0, 2, 1, 3]:
        pytest.skip()

    with pytest.raises(RuntimeError):
        x = wrapper.randu(shape, dtypes.s16)
        y = wrapper.randu(shape, dtypes.s16)

        wrapper.dot(x, y, MatProp.NONE, MatProp.NONE)


# dot all tests
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (100,),
        (1000,),
        (10000,),
        (1, 1),
        (10, 10),
        (100, 100),
        (1000, 1000),
        (1, 1, 1),
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (1, 1, 1, 1),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
    ],
)
def test_dot_all_res_float(shape: tuple) -> None:
    """Test if the dot_all product outputs a float scalar value"""
    dtype = dtypes.f32
    shape = (10,)
    x = wrapper.randu(shape, dtype)

    result = wrapper.dot_all(x, x, MatProp.NONE, MatProp.NONE)

    assert isinstance(result, float)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (100,),
        (1000,),
        (10000,),
        (1, 1),
        (10, 10),
        (100, 100),
        (1000, 1000),
        (1, 1, 1),
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (1, 1, 1, 1),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
    ],
)
def test_dot_all_res_complex(shape: tuple) -> None:
    """Test if the dot_all product outputs a complex scalar value"""
    dtype = dtypes.c32
    shape = (10,)
    x = wrapper.randu(shape, dtype)

    result = wrapper.dot_all(x, x, MatProp.NONE, MatProp.NONE)

    assert isinstance(result, complex)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (100,),
        (1000,),
        (10000,),
        (1, 1),
        (10, 10),
        (100, 100),
        (1000, 1000),
        (1, 1, 1),
        (10, 10, 10),
        (100, 100, 100),
        (1000, 1000, 1000),
        (1, 1, 1, 1),
        (10, 10, 10, 10),
        (100, 100, 100, 100),
        (1000, 1000, 1000, 1000),
    ],
)
def test_dot_all_diff_dtype(shape: tuple) -> None:
    """Test if a dot product of arrays of different dtypes is properly handled"""
    with pytest.raises(RuntimeError):
        x = wrapper.randu(shape, dtypes.f32)
        y = wrapper.randu(shape, dtypes.c32)

        wrapper.dot_all(x, y, MatProp.NONE, MatProp.NONE)


@pytest.mark.parametrize(
    "shape",
    [(1,), (10,), (100,), (1000,)],
)
@pytest.mark.parametrize(
    "dtype_index",
    [i for i in range(13)],
)
def test_dot_all_invalid_dtype(shape: tuple, dtype_index: int) -> None:
    """Test if dot_all properly handles improper dtypes"""
    if dtype_index in [12, 0, 2, 1, 3]:
        pytest.skip()

    with pytest.raises(RuntimeError):
        x = wrapper.randu(shape, dtypes.s16)
        y = wrapper.randu(shape, dtypes.s16)

        wrapper.dot_all(x, y, MatProp.NONE, MatProp.NONE)


@pytest.mark.parametrize(
    "shape_pairs",
    [
        [(5, 1), (1, 6)],
        [(10, 10), (9, 10)],
        [(9, 8), (10, 10)],
        [(random.randint(1, 10), 100), (1000, random.randint(1, 10))],
        [(random.randint(1, 10), 100000), (2, random.randint(1, 10))],
    ],
)
def test_dot_all_invalid_shape_comp(shape_pairs: list) -> None:
    """Test if dot_all properly handles an improper shape pair"""
    with pytest.raises(RuntimeError):
        dtype = dtypes.f32
        x = wrapper.randu(shape_pairs[0], dtype)
        y = wrapper.randu(shape_pairs[1], dtype)

        wrapper.dot_all(x, y, MatProp.NONE, MatProp.NONE)


def test_dot_all_empty_vector() -> None:
    """Test if an empty array passed into the dot product is properly handled"""
    with pytest.raises(RuntimeError):
        empty_shape = (0,)
        dtype = dtypes.f32

        x = wrapper.randu(empty_shape, dtype)
        wrapper.dot(x, x, MatProp.NONE, MatProp.NONE)


# gemm tests
@pytest.mark.parametrize(
    "shape_pairs",
    [
        [(random.randint(1, 10), 10), (10, random.randint(1, 10))],
        [(random.randint(1, 10), 100), (100, random.randint(1, 10))],
        [(random.randint(1, 10), 1000), (1000, random.randint(1, 10))],
    ],
)
def test_gemm_correct_shape_2d(shape_pairs: list) -> None:
    """Test if matmul outputs an array with the correct shape given 2d inputs"""
    dtype = dtypes.f32
    x = wrapper.randu(shape_pairs[0], dtype)
    y = wrapper.randu(shape_pairs[1], dtype)

    result_shape = (shape_pairs[0][0], shape_pairs[1][1])
    result = wrapper.gemm(x, y, MatProp.NONE, MatProp.NONE, 1, 1, None)

    assert wrapper.get_dims(result)[0:2] == result_shape


@pytest.mark.parametrize(
    "shape_pairs",
    [
        [(random.randint(1, 10), 1, 2), (1, random.randint(1, 10), 2)],
        [(random.randint(1, 10), 10, 2), (10, random.randint(1, 10), 2)],
        [(random.randint(1, 10), 100, 2), (100, random.randint(1, 10), 2)],
        [(random.randint(1, 10), 1000, 2), (1000, random.randint(1, 10), 2)],
    ],
)
def test_gemm_correct_shape_3d(shape_pairs: list) -> None:
    """Test if matul outpus an array with the correct shape given 3d inputs"""
    dtype = dtypes.f32
    x = wrapper.randu(shape_pairs[0], dtype)
    y = wrapper.randu(shape_pairs[1], dtype)
    result_shape = (shape_pairs[0][0], shape_pairs[1][1], shape_pairs[0][2])

    result = wrapper.gemm(x, y, MatProp.NONE, MatProp.NONE, 1, 1, None)
    assert wrapper.get_dims(result)[0:3] == result_shape


@pytest.mark.parametrize(
    "shape_pairs",
    [
        [(random.randint(1, 10), 1, 2, 2), (1, random.randint(1, 10), 2, 2)],
        [(random.randint(1, 10), 10, 2, 2), (10, random.randint(1, 10), 2, 2)],
        [(random.randint(1, 10), 100, 2, 2), (100, random.randint(1, 10), 2, 2)],
        [(random.randint(1, 10), 1000, 2, 2), (1000, random.randint(1, 10), 2, 2)],
    ],
)
def test_gemm_correct_shape_4d(shape_pairs: list) -> None:
    """Test if matmul outpus an array with the correct shape given 4d inputs"""
    dtype = dtypes.f32
    x = wrapper.randu(shape_pairs[0], dtype)
    y = wrapper.randu(shape_pairs[1], dtype)
    result_shape = (shape_pairs[0][0], shape_pairs[1][1], shape_pairs[0][2], shape_pairs[0][3])

    result = wrapper.gemm(x, y, MatProp.NONE, MatProp.NONE, 1, 1, None)
    assert wrapper.get_dims(result)[0:4] == result_shape


@pytest.mark.parametrize(
    "dtype",
    [dtypes.f32, dtypes.c32, dtypes.f64, dtypes.c64],
)
def test_gemm_correct_dtype(dtype: dtypes.Dtype) -> None:
    """Test if matmul outputs an array with the correct dtype"""
    if dtype in [dtypes.f64, dtypes.c64] and not wrapper.get_dbl_support():
        pytest.skip()

    shape = (100, 100)
    x = wrapper.randu(shape, dtype)
    y = wrapper.randu(shape, dtype)

    result = wrapper.gemm(x, y, MatProp.NONE, MatProp.NONE, 1, 1, None)

    assert dtypes.c_api_value_to_dtype(wrapper.get_type(result)) == dtype


@pytest.mark.parametrize(
    "shape_pairs",
    [
        [(5, 2), (1, 6)],
        [(10, 10), (9, 10)],
        [(9, 8), (10, 10)],
        [(random.randint(1, 10), 100), (1000, random.randint(1, 10))],
        [(random.randint(1, 10), 100000), (2, random.randint(1, 10))],
    ],
)
def test_gemm_invalid_pair(shape_pairs: list) -> None:
    """Test if matmul handles improper shape pairs"""
    with pytest.raises(RuntimeError):
        dtype = dtypes.f32
        x = wrapper.randu(shape_pairs[0], dtype)
        y = wrapper.randu(shape_pairs[1], dtype)

        wrapper.gemm(x, y, MatProp.NONE, MatProp.NONE, 1, 1, None)


def test_gemm_empty_shape() -> None:
    """Test if matmul handles an empty array"""
    with pytest.raises(RuntimeError):
        empty_shape = (0,)
        dtype = dtypes.f32

        x = wrapper.randu(empty_shape, dtype)
        wrapper.gemm(x, x, MatProp.NONE, MatProp.NONE, 1, 1, None)


@pytest.mark.parametrize(
    "dtype_index",
    [i for i in range(13)],
)
def test_gemm_invalid_dtype(dtype_index: int) -> None:
    """Test if matmul handles an array with an invalid dtype - integer, long, short"""
    shape = (random.randint(1, 10), random.randint(1, 10))
    if dtype_index in [12, 0, 2, 1, 3]:
        pytest.skip()

    dtype = dtypes.c_api_value_to_dtype(dtype_index)

    with pytest.raises(TypeError):
        x = wrapper.randu(shape, dtype)
        y = wrapper.randu(shape, dtype)

        wrapper.gemm(x, y, MatProp.NONE, MatProp.NONE, 1, 1, None)


def test_gemm_empty_matrix() -> None:
    """Test if matmul handles an empty array passed in"""
    with pytest.raises(RuntimeError):
        empty_shape = (0,)
        dtype = dtypes.f32

        x = wrapper.randu(empty_shape, dtype)
        wrapper.gemm(x, x, MatProp.NONE, MatProp.NONE, 1, 1, None)


# matmul tests
@pytest.mark.parametrize(
    "shape_pairs",
    [
        [(random.randint(1, 10), 1), (1, random.randint(1, 10))],
        [(random.randint(1, 10), 10), (10, random.randint(1, 10))],
        [(random.randint(1, 10), 100), (100, random.randint(1, 10))],
        [(random.randint(1, 10), 1000), (1000, random.randint(1, 10))],
    ],
)
def test_matmul_correct_shape_2d(shape_pairs: list) -> None:
    """Test if matmul outputs an array with the correct shape given 2d inputs"""
    dtype = dtypes.f32
    x = wrapper.randu(shape_pairs[0], dtype)
    y = wrapper.randu(shape_pairs[1], dtype)

    result_shape = (shape_pairs[0][0], shape_pairs[1][1])
    result = wrapper.matmul(x, y, MatProp.NONE, MatProp.NONE)

    assert wrapper.get_dims(result)[0 : len(shape_pairs[0])] == result_shape  # noqa: E203


@pytest.mark.parametrize(
    "shape_pairs",
    [
        [(random.randint(1, 10), 1, 2), (1, random.randint(1, 10), 2)],
        [(random.randint(1, 10), 10, 2), (10, random.randint(1, 10), 2)],
        [(random.randint(1, 10), 100, 2), (100, random.randint(1, 10), 2)],
        [(random.randint(1, 10), 1000, 2), (1000, random.randint(1, 10), 2)],
    ],
)
def test_matmul_correct_shape_3d(shape_pairs: list) -> None:
    """Test if matul outpus an array with the correct shape given 3d inputs"""
    dtype = dtypes.f32
    x = wrapper.randu(shape_pairs[0], dtype)
    y = wrapper.randu(shape_pairs[1], dtype)
    result_shape = (shape_pairs[0][0], shape_pairs[1][1], shape_pairs[0][2])

    result = wrapper.matmul(x, y, MatProp.NONE, MatProp.NONE)
    assert wrapper.get_dims(result)[0:3] == result_shape


@pytest.mark.parametrize(
    "shape_pairs",
    [
        [(random.randint(1, 10), 1, 2, 2), (1, random.randint(1, 10), 2, 2)],
        [(random.randint(1, 10), 10, 2, 2), (10, random.randint(1, 10), 2, 2)],
        [(random.randint(1, 10), 100, 2, 2), (100, random.randint(1, 10), 2, 2)],
        [(random.randint(1, 10), 1000, 2, 2), (1000, random.randint(1, 10), 2, 2)],
    ],
)
def test_matmul_correct_shape_4d(shape_pairs: list) -> None:
    """Test if matmul outpus an array with the correct shape given 4d inputs"""
    dtype = dtypes.f32
    x = wrapper.randu(shape_pairs[0], dtype)
    y = wrapper.randu(shape_pairs[1], dtype)
    result_shape = (shape_pairs[0][0], shape_pairs[1][1], shape_pairs[0][2], shape_pairs[0][3])

    result = wrapper.matmul(x, y, MatProp.NONE, MatProp.NONE)
    assert wrapper.get_dims(result)[0:4] == result_shape


@pytest.mark.parametrize(
    "dtype",
    [dtypes.f16, dtypes.f32, dtypes.c32, dtypes.f64, dtypes.c64],
)
def test_matmul_correct_dtype(dtype: dtypes.Dtype) -> None:
    """Test if matmul outputs an array with the correct dtype"""
    if dtype in [dtypes.f64, dtypes.c64] and not wrapper.get_dbl_support():
        pytest.skip()

    shape = (100, 100)
    x = wrapper.randu(shape, dtype)
    y = wrapper.randu(shape, dtype)

    result = wrapper.matmul(x, y, MatProp.NONE, MatProp.NONE)

    assert dtypes.c_api_value_to_dtype(wrapper.get_type(result)) == dtype


@pytest.mark.parametrize(
    "shape_pairs",
    [
        [(5, 2), (1, 6)],
        [(10, 10), (9, 10)],
        [(9, 8), (10, 10)],
        [(random.randint(1, 10), 100), (1000, random.randint(1, 10))],
        [(random.randint(1, 10), 100000), (2, random.randint(1, 10))],
    ],
)
def test_matmul_invalid_pair(shape_pairs: list) -> None:
    """Test if matmul handles improper shape pairs"""
    with pytest.raises(RuntimeError):
        dtype = dtypes.f32
        x = wrapper.randu(shape_pairs[0], dtype)
        y = wrapper.randu(shape_pairs[1], dtype)

        wrapper.matmul(x, y, MatProp.NONE, MatProp.NONE)


def test_matmul_empty_shape() -> None:
    """Test if matmul handles an empty array"""
    with pytest.raises(RuntimeError):
        empty_shape = (0,)
        dtype = dtypes.f32

        x = wrapper.randu(empty_shape, dtype)
        wrapper.matmul(x, x, MatProp.NONE, MatProp.NONE)


@pytest.mark.parametrize(
    "dtype_index",
    [i for i in range(13)],
)
def test_matmul_invalid_dtype(dtype_index: int) -> None:
    """Test if matmul handles an array with an invalid dtype - integer, long, short"""
    shape = (random.randint(1, 10), random.randint(1, 10))
    if dtype_index in [12, 0, 2, 1, 3]:
        pytest.skip()

    dtype = dtypes.c_api_value_to_dtype(dtype_index)

    with pytest.raises(RuntimeError):
        x = wrapper.randu(shape, dtype)
        y = wrapper.randu(shape, dtype)

        wrapper.matmul(x, y, MatProp.NONE, MatProp.NONE)


def test_matmul_empty_matrix() -> None:
    """Test if matmul handles an empty array passed in"""
    with pytest.raises(RuntimeError):
        empty_shape = (0,)
        dtype = dtypes.f32

        x = wrapper.randu(empty_shape, dtype)
        wrapper.matmul(x, x, MatProp.NONE, MatProp.NONE)
