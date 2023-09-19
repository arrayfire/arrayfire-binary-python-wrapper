import ctypes

from arrayfire_wrapper.backend import _backend
from arrayfire_wrapper.defines import AFArray

from ..._error_handler import safe_call


def upper(arr: AFArray, is_unit_diag: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__upper.htm#ga7a4077d52aa4b4b22cccb29a1bdd32ea
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_upper(ctypes.pointer(out), arr, ctypes.c_bool(is_unit_diag)))
    return out
