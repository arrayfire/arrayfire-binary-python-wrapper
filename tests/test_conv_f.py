import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
import arrayfire_wrapper.lib.signal_processing.convolutions as convolutions


# Parameterization for input shapes
@pytest.mark.parametrize(
    "inputShape",
    [
        (7, 7),
        (8, 8),
        (9, 9),
        (10, 10),
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
def test_convolve2_gradient_data(inputShape: tuple[int, int], dtypes: dtype) -> None:
    """Test if convolve gradient returns the correct shape with varying data type and grad type."""
    original_signal = wrapper.randu(inputShape, dtypes)
    original_kernel = wrapper.randu((3, 3), dtypes)
    strides = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    result = convolutions.convolve2_nn(original_signal, original_kernel, strides, padding, dilation)

    match = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    assert inputShape == match, f"Failed for input shape: {inputShape}, Failed for dtype: {dtypes}"


# Third Parameterization for invalid dtypes
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
        original_signal = wrapper.randu((8, 8), invdtypes)
        original_kernel = wrapper.randu((3, 3), invdtypes)
        padding = (1, 1)
        strides = (1, 1)
        dilation = (1, 1)
        result = convolutions.convolve2_nn(original_signal, original_kernel, strides, padding, dilation)
        match = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
        assert match == result, f"Failed for dtype: {invdtypes}"


@pytest.mark.parametrize("padding", [(0, 0), (1, 1), (2, 2)])
def test_convolve2_gradient_padding_variation(padding: tuple[int, int]) -> None:
    """Test convolution with varying padding sizes."""
    original_signal = wrapper.randu((8, 8), dtype.float32)
    original_kernel = wrapper.randu((3, 3), dtype.float32)
    strides = (1, 1)
    dilation = (1, 1)

    result = convolutions.convolve2_nn(original_signal, original_kernel, strides, padding, dilation)

    expected_height = ((5 + 2 * padding[0]) // strides[0]) + 1
    expected_width = ((5 + 2 * padding[1]) // strides[1]) + 1
    expected_shape = (expected_height, expected_width)

    match = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    assert match == expected_shape, f"Failed for padding: {padding}"


@pytest.mark.parametrize("strides", [(1, 1), (2, 2), (3, 3)])
def test_convolve2_gradient_strides_variation(strides: tuple[int, int]) -> None:
    """Test convolution with varying stride sizes."""
    original_signal = wrapper.randu((8, 8), dtype.float32)
    original_kernel = wrapper.randu((3, 3), dtype.float32)
    padding = (1, 1)
    dilation = (1, 1)

    result = convolutions.convolve2_nn(original_signal, original_kernel, strides, padding, dilation)

    expected_height = ((5 + 2 * padding[0]) // strides[0]) + 1
    expected_width = ((5 + 2 * padding[1]) // strides[1]) + 1
    expected_shape = (expected_height, expected_width)

    match = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    assert match == expected_shape, f"Failed for stride: {strides}"


# Parameterize for different dilation sizes
@pytest.mark.parametrize("dilation", [(1, 1), (2, 2), (3, 3)])
def test_convolve2_gradient_dilation_variation(dilation: tuple[int, int]) -> None:
    """Test convolution with varying dilation sizes."""
    original_signal = wrapper.randu((8, 8), dtype.float32)
    original_kernel = wrapper.randu((3, 3), dtype.float32)
    strides = (1, 1)
    padding = (1, 1)

    result = convolutions.convolve2_nn(original_signal, original_kernel, strides, padding, dilation)

    # The calculation of the expected shape might need adjustment based on how dilation affects your convolutions
    expected_height = (
        (
            wrapper.get_dims(original_signal)[0]
            + 2 * padding[0]
            - dilation[0] * (wrapper.get_dims(original_kernel)[0] - 1)
            - 1
        )
        // strides[0]
    ) + 1
    expected_width = (
        (
            wrapper.get_dims(original_signal)[1]
            + 2 * padding[1]
            - dilation[1] * (wrapper.get_dims(original_kernel)[1] - 1)
            - 1
        )
        // strides[1]
    ) + 1
    expected_shape = (expected_height, expected_width)

    match = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    assert match == expected_shape, f"Failed for dilation: {dilation}"


@pytest.mark.parametrize("kernel_size", [(3, 3), (5, 5), (7, 7)])
def test_convolve2_gradient_kernel_size_variation(kernel_size: tuple[int, int]) -> None:
    """Test convolution with varying kernel sizes."""
    original_signal = wrapper.randu((8, 8), dtype.float32)
    original_kernel = wrapper.randu(kernel_size, dtype.float32)
    strides = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)

    result = convolutions.convolve2_nn(original_signal, original_kernel, strides, padding, dilation)

    expected_height = (
        (
            wrapper.get_dims(original_signal)[0]
            + 2 * padding[0]
            - dilation[0] * (wrapper.get_dims(original_kernel)[0] - 1)
            - 1
        )
        // strides[0]
    ) + 1
    expected_width = (
        (
            wrapper.get_dims(original_signal)[1]
            + 2 * padding[1]
            - dilation[1] * (wrapper.get_dims(original_kernel)[1] - 1)
            - 1
        )
        // strides[1]
    ) + 1
    expected_shape = (expected_height, expected_width)

    match = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    assert match == expected_shape, f"Failed for kernel size: {kernel_size}"
