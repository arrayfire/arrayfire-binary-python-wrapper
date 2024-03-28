import pytest

import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.dtypes import Dtype, b8, c32, c64, f16, f32, f64, s16, s32, s64, u8, u16, u32, u64


def check_type_supported(dtype: Dtype) -> None:
    """Checks to see if the specified type is supported by the current system"""
    if dtype in [f64, c64] and not wrapper.get_dbl_support():
        pytest.skip("Device does not support double types")
    if dtype == f16 and not wrapper.get_half_support():
        pytest.skip("Device does not support half types.")


def get_complex_types() -> list:
    """Returns all complex types"""
    return [c32, c64]


def get_real_types() -> list:
    """Returns all real types"""
    return [s16, s32, s64, u8, u16, u32, u64, f16, f32, f64]


def get_all_types() -> list:
    """Returns all types"""
    return [b8, s16, s32, s64, u8, u16, u32, u64, f16, f32, f64, c32, c64]


def get_float_types() -> list:
    """Returns all types"""
    return [f16, f32, f64]
