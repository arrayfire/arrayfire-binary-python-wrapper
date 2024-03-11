import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.lib._constants import ConvGradient
from arrayfire_wrapper.lib.create_and_modify_array.helper_functions import array_to_string


# First parameterization for grad_types
@pytest.mark.parametrize(
    "grad_type",
    [
        0,  # ConvGradient.DEFAULT
        1,  # ConvGradient.FILTER
        2,  # ConvGradient.DATA
        3,  # ConvGradient.BIAS
    ],
)

# Second parameterization for dtypes
@pytest.mark.parametrize(
    "dtypes",
    [
        dtype.float16,  # Floating point 16-bit
        dtype.float32,  # Floating point 32-bit
        dtype.float64,  # Floating point 64-bit
    ],
)
def test_convolve2_gradient_data(grad_type: int, dtypes: dtype) -> None:
    """Test if convolve gradient returns the correct shape with varying data type and grad type."""
    incoming_gradient = wrapper.randu((8, 8), dtypes)
    original_signal = wrapper.randu((10, 10), dtypes)
    original_filter = wrapper.randu((3, 3), dtypes)
    convolved_output = wrapper.randu((8, 8), dtypes)
    strides = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    grad_type_enum = ConvGradient(grad_type)

    result = wrapper.convolve2_gradient_nn(
        incoming_gradient,
        original_signal,
        original_filter,
        convolved_output,
        strides,
        padding,
        dilation,
        grad_type_enum,
    )

    expected_shape = (10, 10, 1, 1) if grad_type != 1 else (3, 3, 1, 1)
    assert wrapper.get_dims(result) == expected_shape, f"Failed for grad_type: {grad_type_enum}, dtype: {dtypes}"


# Third parameterization for dtypes
@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.int32,  # Integer 32-bit
        dtype.uint32,  # Unsigned Integer 32-bit
        dtype.complex32,  # Complex number with float 32-bit real and imaginary
    ],
)
def test_convolve2_gradient_invalid_data(invdtypes: dtype) -> None:
    """Test if convolve gradient returns the correct shape with varying data type and grad type."""
    with pytest.raises(RuntimeError):
        incoming_gradient = wrapper.randu((8, 8), invdtypes)
        original_signal = wrapper.randu((10, 10), invdtypes)
        original_filter = wrapper.randu((3, 3), invdtypes)
        convolved_output = wrapper.randu((8, 8), invdtypes)
        strides = (1, 1)
        padding = (1, 1)
        dilation = (1, 1)
        grad_type_enum = ConvGradient(0)

        result = wrapper.convolve2_gradient_nn(
            incoming_gradient,
            original_signal,
            original_filter,
            convolved_output,
            strides,
            padding,
            dilation,
            grad_type_enum,
        )

        expected_shape = (10, 10, 1, 1)
        assert wrapper.get_dims(result) == expected_shape, f"Failed for dtype: {invdtypes}"


# Parameterization for input shapes
@pytest.mark.parametrize(
    "inputShape",
    [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    ],
)
def test_convolve2_gradient_input(inputShape: tuple[int, int]) -> None:
    """Test if convolve gradient returns the correct shape."""
    incoming_gradient = wrapper.randu((8, 8), dtype.f32)
    original_signal = wrapper.randu(inputShape, dtype.f32)
    original_filter = wrapper.randu((3, 3), dtype.f32)
    convolved_output = wrapper.randu((8, 8), dtype.f32)
    strides = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    grad_type = ConvGradient(0)

    result = wrapper.convolve2_gradient_nn(
        incoming_gradient, original_signal, original_filter, convolved_output, strides, padding, dilation, grad_type
    )
    exp = "Input Shape"
    precision = 4
    transpose = False
    # print(array_to_string(exp, result, precision, transpose))
    match = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    # print(match)
    assert inputShape == match, f"Failed for input shape: {inputShape}"
