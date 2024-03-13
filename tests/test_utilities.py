import arrayfire_wrapper.lib as wrapper
import pytest

from arrayfire_wrapper.dtypes import f64, f16, c32, c64, Dtype


def is_cmplx_type(dtype: Dtype) -> bool:
    """Checks to see if the specified type is a complex type"""
    return dtype == c32 or dtype == c64


def check_type_supported(dtype: Dtype) -> None:
    """Checks to see if the specified type is supported by the current system"""
    if dtype in [f64, c64] and not wrapper.get_dbl_support():
        pytest.skip("Device does not support doubel types.")

    if dtype == f16 and not wrapper.get_half_support():
        pytest.skip("Device does not support half types.")

