import numpy as np
import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
import arrayfire_wrapper.lib.mathematical_functions as ops
from arrayfire_wrapper.lib.create_and_modify_array.helper_functions import array_to_string


import random

@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10), ),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
def test_add_shapes(shape: tuple) -> None:
    """Test addition operation between two arrays of the same shape"""
    lhs = wrapper.randu(shape, dtype.f16)
    rhs = wrapper.randu(shape, dtype.f16)

    result = wrapper.add(lhs, rhs)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape

def test_add_different_shapes() -> None:
    """Test if addition handles arrays of different shapes"""
    with pytest.raises(RuntimeError):
        lhs_shape = (2, 3)
        rhs_shape = (3, 2)
        dtypes = dtype.f16
        lhs = wrapper.randu(lhs_shape, dtypes)
        rhs = wrapper.randu(rhs_shape, dtypes)

        wrapper.add(lhs, rhs)
    
# @pytest.mark.parametrize(
#     "dtype_index",
#     [i for i in range(13)],
# )
# def test_add_dtype(dtype_index: int) -> None:
#     """Test if addition results in an array with the correct dtype"""
#     if (dtype_index in [1, 4]) or (dtype_index in [2, 3] and not wrapper.get_dbl_support()):
#         pytest.skip()

#     shape = (5, 5)
#     dtypes = dtype.c_api_value_to_dtype(dtype_index)
#     lhs = wrapper.randu(shape, dtypes)
#     rhs = wrapper.randu(shape, dtypes)

#     result = wrapper.add(lhs, rhs)

#     assert dtype.c_api_value_to_dtype(wrapper.get_type(result)) == dtypes

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
    'float64': dtype.f64,
    'complex64': dtype.c64,
    'complex32': dtype.c32,
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
    'f64': dtype.f64,
    'c32': dtype.c32,
    'c64': dtype.c64,
    'b8': dtype.b8,
}

@pytest.mark.parametrize("dtype_name", dtype_map.values())
def test_add_supported_dtypes(dtype_name: str) -> None:
    """Test addition operation across all supported data types."""
    shape = (5, 5)  # Using a common shape for simplicity
    lhs = wrapper.randu(shape, dtype_name)
    rhs = wrapper.randu(shape, dtype_name)

    # Performing the addition operation
    result = wrapper.add(lhs, rhs)

    # Checking if the result has the same shape and type as the inputs
    print(dtype_name)
    assert dtype.c_api_value_to_dtype(wrapper.get_type(result)) == dtype_name, f"Failed for dtype: {dtype_name}"

    # # Optionally, perform more specific checks for complex and boolean data types
    # if dtype in [dtypes.c32, dtypes.c64]:
    #     assert isinstance(wrapper.get_data(result), complex), "Result should be of complex type."
    # elif dtype == dtypes.b8:
    #     assert result.dtype == bool, "Result should be of boolean type."