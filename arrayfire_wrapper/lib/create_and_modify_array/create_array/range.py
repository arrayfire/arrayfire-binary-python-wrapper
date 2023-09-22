import ctypes

from arrayfire_wrapper._backend import _backend
from arrayfire_wrapper.defines import AFArray, CShape
from arrayfire_wrapper.dtypes import Dtype
from arrayfire_wrapper.lib._error_handler import safe_call


def range(shape: tuple[int, ...], dim: int, dtype: Dtype, /) -> AFArray:
    """
    source:https://arrayfire.org/docs/group__data__func__range.htm#gadd6c9b479692454670a51e00ea5b26d5
    """
    out = AFArray.create_null_pointer()
    c_shape = CShape(*shape)
    safe_call(_backend.clib.af_range(ctypes.pointer(out), 4, c_shape.c_array, dim, dtype.c_api_value))
    return out
