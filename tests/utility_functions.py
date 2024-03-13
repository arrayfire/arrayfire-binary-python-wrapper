import pytest

import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.dtypes import Dtype, c64, f16, f64


def check_type_supported(dtype: Dtype) -> None:
    """Checks to see if the specified type is supported by the current system"""
    if dtype in [f64, c64] and not wrapper.get_dbl_support():
        pytest.skip("Device does not support double types")

    if dtype == f16 and not wrapper.get_half_support():
        pytest.skip("Device does not support half types.")
