import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.lib._constants import ConvGradient
from arrayfire_wrapper.lib.create_and_modify_array.helper_functions import array_to_string


@pytest.mark.parametrize(
    "grad_types",
    [
        0,
        1,
        2,
        3,
    ],
    "dtypes",
)
def test_convolve2_gradient_nn(grad_types: int, dtypes: dtype) -> None:
    """Test if convolve2_gradient_nn returns the correct shape."""
    incoming_gradient = wrapper.randu((8, 8), dtype.f32)
    original_signal = wrapper.randu((10, 10), dtype.f32)
    original_filter = wrapper.randu((3, 3), dtype.f32)
    convolved_output = wrapper.randu((8, 8), dtype.f32)
    strides = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    grad_type = ConvGradient(grad_types)

    result = wrapper.convolve2_gradient_nn(
        incoming_gradient, original_signal, original_filter, convolved_output, strides, padding, dilation, grad_type
    )
    exp = "Gradient Result"
    precision = 4
    transpose = False
    # print(array_to_string(exp, result, precision, transpose))
    expected_shape = (10, 10, 1, 1)
    if grad_types == 1:
        expected_shape = (3, 3, 1, 1)
    assert wrapper.get_dims(result) == expected_shape, f"Failed for grad_type: {grad_type}"

@pytest.mark.parametrize(
    "inputs",
    [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    ],
)
def test_convolve2_gradient_nn(inputs: tuple[int, int]) -> None:
    """Test if convolve2_gradient_nn returns the correct shape."""
    incoming_gradient = wrapper.randu((8, 8), dtype.f32)
    original_signal = wrapper.randu((10, 10), dtype.f32)
    original_filter = wrapper.randu((3, 3), dtype.f32)
    convolved_output = wrapper.randu((8, 8), dtype.f32)
    strides = inputs
    padding = (1, 1)
    dilation = (1, 1)
    grad_type = ConvGradient(0)

    result = wrapper.convolve2_gradient_nn(
        incoming_gradient, original_signal, original_filter, convolved_output, strides, padding, dilation, grad_type
    )
    exp = "Padding Result"
    precision = 4
    transpose = False
    print(array_to_string(exp, result, precision, transpose))
    match = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    print(match)
    assert inputs == match, f"Failed for input: {inputs}"

    #variable names