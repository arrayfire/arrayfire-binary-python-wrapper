import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def lookup(arr: AFArray, indices: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__index__func__assign.htm#ga93cd5199c647dce0e3b823f063b352ae
    """
    out = AFArray(0)
    call_from_clib(lookup.__name__, ctypes.pointer(out), arr, indices, ctypes.c_int(dim))
    return out
