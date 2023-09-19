import ctypes

from arrayfire_wrapper.backend import _backend
from arrayfire_wrapper.defines import AFArray

from ..._error_handler import safe_call


def diag_create(arr: AFArray, num: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__diag.htm#gaecc9950acc89aefcb99ad805af8aa29b
    """
    out = AFArray(0)
    safe_call(_backend.clib.af_diag_create(ctypes.pointer(out), arr, ctypes.c_int(num)))
    return out


def diag_extract(arr: AFArray, num: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__diag.htm#ga0a28a19534f3c92f11373d662c183061
    """
    out = AFArray(0)
    safe_call(_backend.clib.af_diag_extract(ctypes.pointer(out), arr, ctypes.c_int(num)))
    return out
