import pytest

import arrayfire_wrapper.lib as wrapper

from arrayfire_wrapper.dtypes import f64, c32, c64, Dtype

def is_cmplx_type(dtype: Dtype) -> bool:
    """Checks to see if the specified type is a complex type"""
    return dtype == c32 or dtype == c64

def is_system_supported(dtype: Dtype) -> bool:
    """Checks to see if the specified type is supported by the current system"""
    if dtype in [f64, c64] and not wrapper.get_dbl_support():
        return False

    return True