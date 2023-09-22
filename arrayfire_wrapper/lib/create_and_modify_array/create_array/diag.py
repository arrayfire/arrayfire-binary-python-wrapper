import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def diag_create(arr: AFArray, num: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__diag.htm#gaecc9950acc89aefcb99ad805af8aa29b
    """
    out = AFArray.create_null_pointer()
    call_from_clib(diag_create.__name__, ctypes.pointer(out), arr, ctypes.c_int(num))
    return out


def diag_extract(arr: AFArray, num: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__diag.htm#ga0a28a19534f3c92f11373d662c183061
    """
    out = AFArray(0)
    call_from_clib(diag_extract.__name__, ctypes.pointer(out), arr, ctypes.c_int(num))
    return out
