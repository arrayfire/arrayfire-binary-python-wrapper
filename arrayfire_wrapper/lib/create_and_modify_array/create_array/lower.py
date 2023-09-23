import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def lower(arr: AFArray, is_unit_diag: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__lower.htm#ga40302d71a619692513a4623a89334b52
    """
    out = AFArray.create_null_pointer()
    call_from_clib(lower.__name__, ctypes.pointer(out), arr, ctypes.c_bool(is_unit_diag))
    return out
