import ctypes

from arrayfire_wrapper.defines import AFArray, CShape
from arrayfire_wrapper.dtypes import Dtype
from arrayfire_wrapper.lib._utility import call_from_clib


def identity(shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__identity.htm#gabd9b3de6813a9071ff2c6bc34a65589e
    """
    out = AFArray.create_null_pointer()
    c_shape = CShape(*shape)
    call_from_clib(identity.__name__, ctypes.pointer(out), 4, c_shape.c_array, dtype.c_api_value)
    return out
