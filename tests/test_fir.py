import pytest

import numpy as np

import arrayfire_wrapper.dtypes as dtype
import arrayfire_wrapper.lib as wrapper
import arrayfire_wrapper.lib.signal_processing.filter as fir

@pytest.mark.parametrize("coeff_size", [2, 3, 4])
def test_fir_varying_coefficients(coeff_size):
    """Test FIR filter with varying sizes of the coefficient array."""
    b = wrapper.constant(1, (coeff_size,), dtype.float32)
    x = wrapper.randu((1, 2, 3, 4), dtype.float32)
    wrapper.eval(b)
    wrapper.eval(x)
    b = wrapper.div(b, coeff_size)
    wrapper.eval(b)

    y_af = fir.fir(b, x)

    b_np = np.ones(coeff_size) / coeff_size
    x_np = np.array((1, 2, 3, 4))
    expected_y = np.convolve(x_np, b_np, 'full')[:len(x_np)]

    y = np.array(y_af)
    np.testing.assert_almost_equal(y, expected_y, decimal=5, err_msg=f"FIR filter operation failed for coeff_size: {coeff_size}")
