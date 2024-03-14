import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
import arrayfire_wrapper.lib.signal_processing.convolutions as convolutions
from arrayfire_wrapper.lib._constants import ConvDomain, ConvMode


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
def test_convolve2_gradient_data(inputShape: tuple[int, int], dtypes: dtype.Dtype) -> None:
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
def test_convolve2_nn_invalid_data(invdtypes: dtype.Dtype) -> None:
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
def test_convolve2_nn_padding_variation(padding: tuple[int, int]) -> None:
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
def test_convolve2_nn_strides_variation(strides: tuple[int, int]) -> None:
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
def test_convolve2_nn_dilation_variation(dilation: tuple[int, int]) -> None:
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
def test_convolve2_nn_kernel_size_variation(kernel_size: tuple[int, int]) -> None:
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


@pytest.mark.parametrize("input_size", [(8,), (12, 12), (10, 10, 10)])
def test_convolve1_input_size(input_size: tuple[int, int]) -> None:
    """Test convolve1 with varying input sizes."""
    signal = wrapper.randu(input_size, dtype.float32)
    filter = wrapper.randu((3,), dtype.float32)

    result = convolutions.convolve1(signal, filter, ConvMode(0), ConvDomain(0))

    expected_output_size = wrapper.get_dims(signal)[0]
    assert wrapper.get_dims(result)[0] == expected_output_size, f"Failed for input_size: {input_size}"


@pytest.mark.parametrize("filter_size", [(3,), (5,), (7,)])
def test_convolve1_filter_size(filter_size: tuple[int, int]) -> None:
    """Test convolve1 with varying filter sizes."""
    signal = wrapper.randu((10, 10), dtype.float32)
    filter = wrapper.randu(filter_size, dtype.float32)

    result = convolutions.convolve1(signal, filter, ConvMode(0), ConvDomain(0))

    # Assuming ConvMode(0) maintains the input size
    expected_output_size = wrapper.get_dims(signal)[0]
    assert wrapper.get_dims(result)[0] == expected_output_size, f"Failed for filter_size: {filter_size}"


@pytest.mark.parametrize("conv_mode", [0, 1])  # 0: AF_CONV_DEFAULT, 1: AF_CONV_EXPAND
def test_convolve1_conv_mode(conv_mode: int) -> None:
    """Test convolve1 with varying convolution modes."""
    input_size = 10
    filter_size = 3
    signal = wrapper.randu((input_size, input_size), dtype.float32)
    filter = wrapper.randu((filter_size,), dtype.float32)

    result = convolutions.convolve1(signal, filter, ConvMode(conv_mode), ConvDomain(0))

    if conv_mode == 0:  # AF_CONV_DEFAULT
        expected_output_size = input_size
    elif conv_mode == 1:  # AF_CONV_EXPAND
        expected_output_size = input_size + filter_size - 1

    assert wrapper.get_dims(result)[0] == expected_output_size, f"Failed for conv_mode: {conv_mode}"


@pytest.mark.parametrize("conv_domain", [0, 1, 2])  # AUTO, SPATIAL, and FREQ
def test_convolve1_conv_domain(conv_domain: int) -> None:
    """Test convolve1 with varying convolution domains."""
    input_size = 10
    filter_size = 3
    signal = wrapper.randu((input_size, input_size), dtype.float32)
    filter = wrapper.randu((filter_size,), dtype.float32)

    result = convolutions.convolve1(signal, filter, ConvMode(0), ConvDomain(conv_domain))

    assert wrapper.get_dims(result)[0] == input_size, f"Failed for conv_domain: {ConvDomain(conv_domain)}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.int32,  # Integer 32-bit
        dtype.uint32,  # Unsigned Integer 32-bit
        dtype.complex32,  # Complex number with float 32-bit real and imaginary
    ],
)
def test_convolve1_valid(invdtypes: dtype.Dtype) -> None:
    """Test convolve1 with valid dtypes."""
    input_size = 10
    filter_size = 3
    signal = wrapper.randu((input_size, input_size), invdtypes)
    filter = wrapper.randu((filter_size,), invdtypes)

    result = convolutions.convolve1(signal, filter, ConvMode(0), ConvDomain(0))

    assert wrapper.get_dims(result)[0] == input_size, f"Failed for dtype: {invdtypes}"


@pytest.mark.parametrize("input_size", [(8,), (12, 12), (10, 10, 10)])
def test_fft_convolve1_input_size(input_size: tuple[int, int]) -> None:
    """Test fft_convolve1 with varying input sizes."""
    signal = wrapper.randu(input_size, dtype.float32)
    filter = wrapper.randu((3,), dtype.float32)

    result = convolutions.fft_convolve1(signal, filter, ConvMode(0))

    expected_output_size = wrapper.get_dims(result)[0]
    assert wrapper.get_dims(result)[0] == expected_output_size, f"Failed for input_size: {input_size}"


@pytest.mark.parametrize("filter_size", [(3,), (5,), (7,)])
def test_fft_convolve1_filter_size(filter_size: tuple[int, int]) -> None:
    """Test fft_convolve1 with varying filter sizes."""
    signal = wrapper.randu((10, 10), dtype.float32)
    filter = wrapper.randu(filter_size, dtype.float32)

    result = convolutions.fft_convolve1(signal, filter, ConvMode(0))

    expected_output_size = wrapper.get_dims(signal)[0]
    assert wrapper.get_dims(result)[0] == expected_output_size, f"Failed for filter_size: {filter_size}"


@pytest.mark.parametrize("conv_mode", [0, 1])
def test_fft_convolve1_conv_mode(conv_mode: int) -> None:
    """Test fft_convolve1 with varying convolution modes."""
    input_size = (10, 10)
    filter_size = 3
    signal = wrapper.randu(input_size, dtype.float32)
    filter = wrapper.randu((filter_size,), dtype.float32)

    result = convolutions.fft_convolve1(signal, filter, ConvMode(conv_mode))

    if conv_mode == 0:  # AF_CONV_DEFAULT
        expected_output_size = input_size[0]
    elif conv_mode == 1:  # AF_CONV_EXPAND
        expected_output_size = input_size[0] + filter_size - 1
    assert wrapper.get_dims(result)[0] == expected_output_size, f"Failed for conv_mode: {conv_mode}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.int32,  # Integer 32-bit
        dtype.uint32,  # Unsigned Integer 32-bit
        dtype.complex32,  # Complex number with float 32-bit real and imaginary
    ],
)
def test_fft_convolve1_valid(invdtypes: dtype.Dtype) -> None:
    """Test fft_convolve1 with valid dtypes."""
    signal = wrapper.randu((10, 10), invdtypes)
    filter = wrapper.randu((3,), invdtypes)

    result = convolutions.fft_convolve1(signal, filter, ConvMode(0))

    expected_output_size = wrapper.get_dims(signal)[0]
    assert wrapper.get_dims(result)[0] == expected_output_size, f"Failed for dtype: {invdtypes}"


@pytest.mark.parametrize("input_size", [(8, 8), (12, 12, 12), (10, 10)])
def test_convolve2_input_size(input_size: tuple[int, int]) -> None:
    """Test convolve2 with varying input sizes."""
    signal = wrapper.randu(input_size, dtype.float32)
    filter = wrapper.randu((3, 3), dtype.float32)

    result = convolutions.convolve2(signal, filter, ConvMode(0), ConvDomain(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
    ) == expected_output, f"Failed for input_size: {input_size}"


@pytest.mark.parametrize("filter_size", [(3, 3), (5,), (7, 7, 7)])
def test_convolve2_filter_size(filter_size: tuple[int, int]) -> None:
    """Test convolve2 with varying filter sizes."""
    signal = wrapper.randu((10, 10), dtype.float32)
    filter = wrapper.randu(filter_size, dtype.float32)

    result = convolutions.convolve2(signal, filter, ConvMode(0), ConvDomain(0))

    expected_output = (wrapper.get_dims(signal)[0], wrapper.get_dims(signal)[1])
    assert (
        wrapper.get_dims(result)[0],
        wrapper.get_dims(result)[1],
    ) == expected_output, f"Failed for filter_size: {filter_size}"


@pytest.mark.parametrize("conv_mode", [0, 1])  # 0: Default, 1: Expand
def test_convolve2_conv_mode(conv_mode: int) -> None:
    """Test convolve2 with varying convolution modes."""
    signal = wrapper.randu((10, 10), dtype.float32)
    filter = wrapper.randu((3, 3), dtype.float32)

    result = convolutions.convolve2(signal, filter, ConvMode(conv_mode), ConvDomain(0))

    if conv_mode == 0:  # Default
        expected_output = (10, 10)
    else:  # Expand
        expected_output = (12, 12)

    assert (
        wrapper.get_dims(result)[0],
        wrapper.get_dims(result)[1],
    ) == expected_output, f"Failed for conv_mode: {conv_mode}"


@pytest.mark.parametrize("conv_domain", [0, 1, 2])  # 0: Auto, 1: Spatial, 2: Frequency
def test_convolve2_conv_domain(conv_domain: int) -> None:
    """Test convolve2 with varying convolution domains."""
    signal = wrapper.randu((10, 10), dtype.float32)
    filter = wrapper.randu((3, 3), dtype.float32)

    result = convolutions.convolve2(signal, filter, ConvMode(0), ConvDomain(conv_domain))

    expected_output = (10, 10)

    assert (
        wrapper.get_dims(result)[0],
        wrapper.get_dims(result)[1],
    ) == expected_output, f"Failed for conv_domain: {conv_domain}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.int32,  # Integer 32-bit
        dtype.uint32,  # Unsigned Integer 32-bit
        dtype.complex32,  # Complex number with float 32-bit real and imaginary
    ],
)
def test_convolve2_conv_valid_dtype(invdtypes: dtype.Dtype) -> None:
    """Test fft_convolve1 with valid dtypes."""
    signal = wrapper.randu((10, 10), invdtypes)
    filter = wrapper.randu((3, 3), invdtypes)

    result = convolutions.convolve2(signal, filter, ConvMode(0), ConvDomain(0))

    expected_output_size = wrapper.get_dims(signal)[0]
    assert wrapper.get_dims(result)[0] == expected_output_size, f"Failed for dtype: {invdtypes}"


@pytest.mark.parametrize("input_size", [(8, 8), (12, 12, 12), (10, 10)])
def test_fftConvolve2_input_size(input_size: tuple[int, int]) -> None:
    """Test fftConvolve2 with varying input sizes."""
    signal = wrapper.randu(input_size, dtype.float32)
    filter = wrapper.randu((3, 3), dtype.float32)

    result = convolutions.fft_convolve2(signal, filter, ConvMode(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
    ) == expected_output, f"Failed for input_size: {input_size}"


@pytest.mark.parametrize("filter_size", [(3, 3), (5, 5), (7, 7, 7)])
def test_fftConvolve2_filter_size(filter_size: tuple[int, int]) -> None:
    """Test fftConvolve2 with varying filter sizes."""
    signal = wrapper.randu((10, 10), dtype.float32)
    filter = wrapper.randu(filter_size, dtype.float32)

    result = convolutions.fft_convolve2(signal, filter, ConvMode(0))

    expected_output = (wrapper.get_dims(signal)[0], wrapper.get_dims(signal)[1])
    assert (
        wrapper.get_dims(result)[0],
        wrapper.get_dims(result)[1],
    ) == expected_output, f"Failed for filter_size: {filter_size}"


@pytest.mark.parametrize("conv_mode", [0, 1])  # 0: Default, 1: Expand
def test_fftConvolve2_conv_mode(conv_mode: int) -> None:
    """Test fftConvolve2 with varying convolution modes."""
    signal = wrapper.randu((10, 10), dtype.float32)
    filter = wrapper.randu((3, 3), dtype.float32)

    result = convolutions.fft_convolve2(signal, filter, ConvMode(conv_mode))

    if conv_mode == 0:  # Default
        expected_output = (10, 10)
    else:  # Expand
        expected_output = (12, 12)

    assert (
        wrapper.get_dims(result)[0],
        wrapper.get_dims(result)[1],
    ) == expected_output, f"Failed for conv_mode: {conv_mode}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.int32,  # Integer 32-bit
        dtype.uint32,  # Unsigned Integer 32-bit
        dtype.complex32,  # Complex number with float 32-bit real and imaginary
        dtype.bool,
    ],
)
def test_fftConvolve2_valid_dtype(invdtypes: dtype.Dtype) -> None:
    """Test fft_convolve1 with valid dtypes."""
    signal = wrapper.randu((10, 10), invdtypes)
    filter = wrapper.randu((3, 3), invdtypes)

    result = convolutions.fft_convolve2(signal, filter, ConvMode(0))

    expected_output = (wrapper.get_dims(signal)[0], wrapper.get_dims(signal)[1])
    assert (
        wrapper.get_dims(result)[0],
        wrapper.get_dims(result)[1],
    ) == expected_output, f"Failed for dtype: {invdtypes}"


@pytest.mark.parametrize("input_size", [(8, 8), (12, 12, 12), (10, 10)])
def test_convolve2_sep_input_size(input_size: tuple[int, int]) -> None:
    """Test convolve2_sep with varying input sizes."""
    signal = wrapper.randu(input_size, dtype.float32)
    col_filter = wrapper.randu((3, 1), dtype.float32)
    row_filter = wrapper.randu((3, 1), dtype.float32)

    result = convolutions.convolve2_sep(col_filter, row_filter, signal, ConvMode(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
    ) == expected_output, f"Failed for input_size: {input_size}"


@pytest.mark.parametrize("col_vector", [(3, 1), (4, 1), (5, 1)])
def test_convolve2_sep_col_vector(col_vector: tuple[int, int]) -> None:
    """Test convolve2_sep with varying column vector sizes."""
    signal = wrapper.randu((8, 8), dtype.float32)
    col_filter = wrapper.randu(col_vector, dtype.float32)
    row_filter = wrapper.randu((3, 1), dtype.float32)

    result = convolutions.convolve2_sep(col_filter, row_filter, signal, ConvMode(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
    ) == expected_output, f"Failed for column vector: {col_vector}"


@pytest.mark.parametrize("row_vector", [(3, 1), (4, 1), (5, 1)])
def test_convolve2_sep_row_vector(row_vector: tuple[int, int]) -> None:
    """Test convolve2_sep with varying row vector sizes."""
    signal = wrapper.randu((8, 8), dtype.float32)
    col_filter = wrapper.randu((3, 1), dtype.float32)
    row_filter = wrapper.randu(row_vector, dtype.float32)

    result = convolutions.convolve2_sep(col_filter, row_filter, signal, ConvMode(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
    ) == expected_output, f"Failed for column vector: {row_vector}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.int32,  # Integer 32-bit
        dtype.uint32,  # Unsigned Integer 32-bit
        dtype.complex32,  # Complex number with float 32-bit real and imaginary
        dtype.bool,
    ],
)
def test_convolve2_valid_dtypes(invdtypes: dtype.Dtype) -> None:
    """Test convolve2_sep with varying invalid data types."""
    signal = wrapper.randu((8, 8), invdtypes)
    col_filter = wrapper.randu((3, 1), invdtypes)
    row_filter = wrapper.randu((3, 1), invdtypes)

    result = convolutions.convolve2_sep(col_filter, row_filter, signal, ConvMode(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
    ) == expected_output, f"Failed for data type: {invdtypes}"


@pytest.mark.parametrize("input_size", [(8, 8, 8), (12, 12, 12), (10, 10, 10, 10)])
def test_convolve3_input_size(input_size: tuple[int, int, int]) -> None:
    """Test convolve3 with varying input sizes."""
    signal = wrapper.randu(input_size, dtype.f32)
    filter = wrapper.randu((3, 3, 3), dtype.f32)

    result = convolutions.convolve3(signal, filter, ConvMode(0), ConvDomain(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1], wrapper.get_dims(result)[2])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
        wrapper.get_dims(signal)[2],
    ) == expected_output, f"Failed for input_size: {input_size}"


@pytest.mark.parametrize(
    "filter_size",
    [
        (3, 3, 3),
        (
            5,
            5,
        ),
        (2, 2, 2, 2),
    ],
)
def test_convolve3_filter_size(filter_size: tuple[int, int, int]) -> None:
    """Test convolve3 with varying filter sizes."""
    signal = wrapper.randu((10, 10, 10), dtype.f32)
    filter = wrapper.randu(filter_size, dtype.f32)

    result = convolutions.convolve3(signal, filter, ConvMode(0), ConvDomain(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1], wrapper.get_dims(result)[2])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
        wrapper.get_dims(signal)[2],
    ) == expected_output, f"Failed for filter_size: {filter_size}"


@pytest.mark.parametrize("conv_mode", [0, 1])  # 0: Default, 1: Expand
def test_convolve3_conv_mode(conv_mode: int) -> None:
    """Test convolve3 with varying convolution modes."""
    signal = wrapper.randu((10, 10, 10), dtype.f32)
    filter = wrapper.randu((3, 3, 3), dtype.f32)

    result = convolutions.convolve3(signal, filter, ConvMode(conv_mode), ConvDomain(0))

    if conv_mode == 0:
        expected_output = (10, 10, 10)
    else:
        expected_output = (10 + 3 - 1, 10 + 3 - 1, 10 + 3 - 1)

    assert (
        wrapper.get_dims(result)[0],
        wrapper.get_dims(result)[1],
        wrapper.get_dims(result)[2],
    ) == expected_output, f"Failed for conv_mode: {conv_mode}"


@pytest.mark.parametrize(
    "invdtypes",
    [
        dtype.int32,  # Integer 32-bit
        dtype.uint32,  # Unsigned Integer 32-bit
        dtype.complex32,  # Complex number with float 32-bit real and imaginary
        dtype.bool,
    ],
)
def test_convolve3_valid_dtype(invdtypes: dtype.Dtype) -> None:
    """Test convolve3 with valid data types."""
    signal = wrapper.randu((10, 10, 10), invdtypes)
    filter = wrapper.randu((3, 3, 3), invdtypes)

    result = convolutions.convolve3(signal, filter, ConvMode(0), ConvDomain(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1], wrapper.get_dims(result)[2])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
        wrapper.get_dims(signal)[2],
    ) == expected_output, f"Failed for dtype: {invdtypes}"


@pytest.mark.parametrize("conv_domain", [0, 1, 2])
def test_convolve3_conv_domain(conv_domain: int) -> None:
    """Test convolve3 with varying convolution domains."""
    signal = wrapper.randu((10, 10, 10), dtype.f32)
    filter = wrapper.randu((3, 3, 3), dtype.f32)

    result = convolutions.convolve3(signal, filter, ConvMode(0), ConvDomain(conv_domain))

    expected_output = (10, 10, 10)

    assert (
        wrapper.get_dims(result)[0],
        wrapper.get_dims(result)[1],
        wrapper.get_dims(result)[2],
    ) == expected_output, f"Failed for conv_mode: {conv_domain}"


@pytest.mark.parametrize("input_size", [(8, 8, 8), (12, 12, 12), (10, 10, 10, 10)])
def test_fft_convolve3_input_size(input_size: tuple[int, int, int]) -> None:
    """Test fft_convolve3 with varying input sizes."""
    signal = wrapper.randu(input_size, dtype.f32)
    filter = wrapper.randu((3, 3, 3), dtype.f32)

    result = wrapper.fft_convolve3(signal, filter, ConvMode(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1], wrapper.get_dims(result)[2])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
        wrapper.get_dims(signal)[2],
    ) == expected_output, f"Failed for input_size: {input_size}"


@pytest.mark.parametrize("filter_size", [(3, 3, 3), (5, 5, 5), (2, 2, 2, 2)])
def test_fft_convolve3_filter_size(filter_size: tuple[int, int, int]) -> None:
    """Test fft_convolve3 with varying filter sizes."""
    signal = wrapper.randu((10, 10, 10), dtype.f32)
    filter = wrapper.randu(filter_size, dtype.f32)

    result = wrapper.fft_convolve3(signal, filter, ConvMode(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1], wrapper.get_dims(result)[2])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
        wrapper.get_dims(signal)[2],
    ) == expected_output, f"Failed for filter_size: {filter_size}"


@pytest.mark.parametrize("conv_mode", [0, 1])  # Assuming 0: Default, 1: Expand, for example purposes
def test_fft_convolve3_conv_mode(conv_mode: int) -> None:
    """Test fft_convolve3 with varying convolution modes."""
    signal = wrapper.randu((10, 10, 10), dtype.f32)
    filter = wrapper.randu((3, 3, 3), dtype.f32)

    result = wrapper.fft_convolve3(signal, filter, ConvMode(conv_mode))
    if conv_mode == 0:
        expected_output = (10, 10, 10)
    else:
        expected_output = (10 + 3 - 1, 10 + 3 - 1, 10 + 3 - 1)

    assert (
        wrapper.get_dims(result)[0],
        wrapper.get_dims(result)[1],
        wrapper.get_dims(result)[2],
    ) == expected_output, f"Failed for conv_mode: {conv_mode}"


@pytest.mark.parametrize(
    "valid_dtype",
    [
        dtype.f32,  # Floating-point 32-bit
        dtype.f64,  # Floating-point 64-bit
        dtype.c32,  # Complex number with float 32-bit real and imaginary
        dtype.bool,  # Typically not supported for FFT convolutions
    ],
)
def test_fft_convolve3_valid_dtype(valid_dtype: dtype.Dtype) -> None:
    """Test fft_convolve3 with valid data types."""
    signal = wrapper.randu((10, 10, 10), valid_dtype)
    filter = wrapper.randu((3, 3, 3), valid_dtype)

    result = wrapper.fft_convolve3(signal, filter, ConvMode(0))

    expected_output = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1], wrapper.get_dims(result)[2])
    assert (
        wrapper.get_dims(signal)[0],
        wrapper.get_dims(signal)[1],
        wrapper.get_dims(signal)[2],
    ) == expected_output, f"Failed for dtype: {valid_dtype}"
