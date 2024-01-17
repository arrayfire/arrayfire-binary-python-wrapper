import ctypes

from arrayfire_wrapper.defines import AFArray, CShape
from arrayfire_wrapper.dtypes import Dtype
from arrayfire_wrapper.lib._utility import call_from_clib


def iota(shape: tuple[int, ...], t_shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__iota.htm#ga032c10a9bbd3cf051f711bfea1dea65c
    """
    out = AFArray.create_null_pointer()
    c_shape = CShape(*shape)
    t_c_shape = CShape(*t_shape)
    call_from_clib(iota.__name__, ctypes.pointer(out), 4, c_shape.c_array, 4, t_c_shape.c_array, dtype.c_api_value)
    return out
