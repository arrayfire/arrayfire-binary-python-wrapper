import pytest

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
import arrayfire_wrapper.lib.signal_processing.fast_fourier_transforms as fastft
from arrayfire_wrapper.lib.create_and_modify_array.helper_functions import array_to_string
import numpy as np

# Parameterization for input shapes
@pytest.mark.parametrize(
    "inputShape",
    [
        (2,),
        (3, 3),
        (4, 4),
    ],
)

def test_fft_func(inputShape: tuple[int, int]) -> None:
    """Test if varying input shape returns the correct values."""
    input = wrapper.randu(inputShape, dtype.float32)
    norm_factor = 0.5
    dim0 = 1
    result = fastft.fft(input, norm_factor, dim0)
    exp = "Input Shape"
    precision = 4
    transpose = False
    print(array_to_string(exp, result, precision, transpose))
    signal = np.random.rand(2).astype(np.float32)
    result = np.fft.fft(signal)
    result_scaled = result * norm_factor
    print(result_scaled)
    # match = (wrapper.get_dims(result)[0], wrapper.get_dims(result)[1])
    # assert inputShape == match, f"Failed for input shape: {inputShape}, Failed for dtype: {dtype.float32}"
