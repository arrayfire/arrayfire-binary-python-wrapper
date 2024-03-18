import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.lib.create_and_modify_array.helper_functions import array_to_string

dtype_map = {
    'int16': dtype.s16,
    'int32': dtype.s32,
    'int64': dtype.s64,
    'uint8': dtype.u8,
    'uint16': dtype.u16,
    'uint32': dtype.u32,
    'uint64': dtype.u64,
    # 'float16': dtype.f16,
    # 'float32': dtype.f32,
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
    # 'f16': dtype.f16,
    # 'f32': dtype.f32,
    # 'f64': dtype.f64,
    # 'c32': dtype.c32,
    # 'c64': dtype.c64,
    'b8': dtype.b8,
}
@pytest.mark.parametrize("dtype_name", dtype_map.values())
def test_bitshiftl_dtypes(dtype_name: dtype.Dtype) -> None:
    """Test bit shift operation across all supported data types."""
    shape = (5, 5)
    values = wrapper.randu(shape, dtype_name)
    bits_to_shift = wrapper.constant(1, shape, dtype_name)
    
    result = wrapper.bitshiftl(values, bits_to_shift)

    assert dtype.c_api_value_to_dtype(wrapper.get_type(result)) == dtype_name, f"Failed for dtype: {dtype_name}"
@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_bitshiftl_supported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test bitshift operations for unsupported integer data types."""
    shape = (5, 5)
    with pytest.raises(RuntimeError):
        value = wrapper.randu(shape, invdtypes)
        shift_amount = 1 

        result = wrapper.bitshiftl(value, shift_amount)
        assert dtype.c_api_value_to_dtype(wrapper.get_type(result)) == invdtypes, f"Failed for dtype: {invdtypes}"
@pytest.mark.parametrize("input_size", [8, 10, 12])
def test_bitshiftl_varying_input_size(input_size):
    """Test bitshift left operation with varying input sizes"""
    shape = (input_size, input_size)
    value = wrapper.randu(shape, dtype.int16)
    shift_amount = wrapper.constant(1, shape, dtype.int16)  # Fixed shift amount for simplicity

    result = wrapper.bitshiftl(value, shift_amount)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape

@pytest.mark.parametrize(
    "shape",
    [
        (10, ),
        (5, 5),
        (2, 3, 4),
    ],
)
def test_bitshiftl_varying_shapes(shape: tuple) -> None:
    """Test left bit shifting with arrays of varying shapes."""
    values = wrapper.randu(shape, dtype.int16)
    bits_to_shift = wrapper.constant(1, shape, dtype.int16)
    
    result = wrapper.bitshiftl(values, bits_to_shift)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape

@pytest.mark.parametrize("shift_amount", [0, 2, 30])
def test_bitshift_left_varying_shift_amount(shift_amount):
    """Test bitshift left operation with varying shift amounts."""
    shape = (5, 5)
    value = wrapper.randu(shape, dtype.int16)
    shift_amount_arr = wrapper.constant(shift_amount, shape, dtype.int16)

    result = wrapper.bitshiftl(value, shift_amount_arr)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape

@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((1, 5), (5, 1)),  # 2D with 2D inverse
        ((5, 5), (5, 1)),  # 2D with 2D
        ((5, 5), (1, 1)),  # 2D with 2D
        ((1, 1, 1), (5, 5, 5)),  # 3D with 3D
    ],
)
def test_bitshiftl_different_shapes(shape_a: tuple, shape_b: tuple) -> None:
    """Test if left bit shifting handles arrays of different shapes"""
    with pytest.raises(RuntimeError):
        values = wrapper.randu(shape_a, dtype.int16)
        bits_to_shift = wrapper.constant(1, shape_b, dtype.int16)
        result = wrapper.bitshiftl(values, bits_to_shift)
        print(array_to_string("", result, 3, False))
        assert wrapper.get_dims(result)[0 : len(shape_a)] == shape_a, f"Failed for shapes {shape_a} and {shape_b}"

@pytest.mark.parametrize("shift_amount", [0, 2, 30])
def test_bitshift_right_varying_shift_amount(shift_amount):
    """Test bitshift right operation with varying shift amounts."""
    shape = (5, 5)
    value = wrapper.randu(shape, dtype.int16)
    shift_amount_arr = wrapper.constant(shift_amount, shape, dtype.int16)

    result = wrapper.bitshiftr(value, shift_amount_arr)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape

@pytest.mark.parametrize("dtype_name", dtype_map.values())
def test_bitshiftr_dtypes(dtype_name: dtype.Dtype) -> None:
    """Test bit shift operation across all supported data types."""
    shape = (5, 5)
    values = wrapper.randu(shape, dtype_name)
    bits_to_shift = wrapper.constant(1, shape, dtype_name)
    
    result = wrapper.bitshiftr(values, bits_to_shift)

    assert dtype.c_api_value_to_dtype(wrapper.get_type(result)) == dtype_name, f"Failed for dtype: {dtype_name}"
@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.c64,
        dtype.f64,
    ],
)
def test_bitshiftr_supported_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test bitshift operations for unsupported integer data types."""
    shape = (5, 5)
    with pytest.raises(RuntimeError):
        value = wrapper.randu(shape, invdtypes)
        shift_amount = 1 

        result = wrapper.bitshiftr(value, shift_amount)
        assert dtype.c_api_value_to_dtype(wrapper.get_type(result)) == invdtypes, f"Failed for dtype: {invdtypes}"

@pytest.mark.parametrize("input_size", [8, 10, 12])
def test_bitshift_right_varying_input_size(input_size):
    """Test bitshift right operation with varying input sizes"""
    shape = (input_size, input_size)
    value = wrapper.randu(shape, dtype.int16)
    shift_amount = wrapper.constant(1, shape, dtype.int16)  # Fixed shift amount for simplicity

    result = wrapper.bitshiftr(value, shift_amount)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape
@pytest.mark.parametrize(
    "shape",
    [
        (10, ),
        (5, 5),
        (2, 3, 4),
    ],
)
def test_bitshiftr_varying_shapes(shape: tuple) -> None:
    """Test right bit shifting with arrays of varying shapes."""
    values = wrapper.randu(shape, dtype.int16)
    bits_to_shift = wrapper.constant(1, shape, dtype.int16)
    
    result = wrapper.bitshiftr(values, bits_to_shift)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape



@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((1, 5), (5, 1)),  # 2D with 2D inverse
        ((5, 5), (5, 1)),  # 2D with 2D
        ((5, 5), (1, 1)),  # 2D with 2D
        ((1, 1, 1), (5, 5, 5)),  # 3D with 3D
    ],
)
def test_bitshiftr_different_shapes(shape_a: tuple, shape_b: tuple) -> None:
    """Test if right bit shifting handles arrays of different shapes"""
    with pytest.raises(RuntimeError):
        values = wrapper.randu(shape_a, dtype.int16)
        bits_to_shift = wrapper.constant(1, shape_b, dtype.int16)
        result = wrapper.bitshiftr(values, bits_to_shift)
        print(array_to_string("", result, 3, False))
        assert wrapper.get_dims(result)[0 : len(shape_a)] == shape_a, f"Failed for shapes {shape_a} and {shape_b}"