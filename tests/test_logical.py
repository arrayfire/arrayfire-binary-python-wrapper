import random

import numpy as np
import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper

from arrayfire_wrapper.lib.create_and_modify_array.helper_functions import array_to_string

dtype_map = {
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
    "complex32": dtype.c32,
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
    "c32": dtype.c32,
    # 'c64': dtype.c64,
    "b8": dtype.b8,
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
def test_and_shape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test and_ operation between two arrays of the same shape"""
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.and_(lhs, rhs)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape} and dtype {dtype_name}"  # noqa

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

        assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape} and dtype {invdtypes}"  # noqa
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
def test_bitandshape_dtypes(shape: tuple, dtype_name: dtype.Dtype) -> None:
    """Test bitand operation between two arrays of the same shape"""
    print(dtype_name)
    if dtype_name == dtype.c32 or dtype_name == dtype.c64 or dtype_name == dtype.f32 or dtype_name == dtype.f64 or dtype_name == dtype.f16:
        pytest.skip()
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    result = wrapper.bitand(lhs, rhs)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape, f"failed for shape: {shape} and dtype {dtype_name}"  # noqa

@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_bitandshapes_invalid(invdtypes: dtype.Dtype) -> None:
    """Test bitand operation between two arrays of the same shape"""
    with pytest.raises(RuntimeError):
        shape = (3, 3)
        lhs = wrapper.randu(shape, invdtypes)
        rhs = wrapper.randu(shape, invdtypes)