import ctypes

from arrayfire_wrapper.backend import _backend
from arrayfire_wrapper.defines import AFArray

from ..._error_handler import safe_call


def lower(arr: AFArray, is_unit_diag: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__lower.htm#ga40302d71a619692513a4623a89334b52
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_lower(ctypes.pointer(out), arr, ctypes.c_bool(is_unit_diag)))
    return out
