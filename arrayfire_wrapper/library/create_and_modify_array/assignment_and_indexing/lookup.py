import ctypes

from arrayfire_wrapper.backend import _backend
from arrayfire_wrapper.defines import AFArray

from ..._error_handler import safe_call


def lookup(arr: AFArray, indices: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__index__func__assign.htm#ga93cd5199c647dce0e3b823f063b352ae
    """
    out = AFArray(0)
    safe_call(_backend.clib.af_assign_gen(ctypes.pointer(out), arr, indices, ctypes.c_int(dim)))
    return out
