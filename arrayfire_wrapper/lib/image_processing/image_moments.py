import ctypes

from arrayfire_wrapper.backend import _backend
from arrayfire_wrapper.defines import AFArray, Moment
from arrayfire_wrapper.lib._error_handler import safe_call


def moments(arr: AFArray, moment: Moment) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__moments.htm#ga28bab821ee673f9a93882e486e8cd47d
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_moments(ctypes.pointer(out), arr, moment.value))
    return out


def moments_all(arr: AFArray, moment: Moment) -> float:
    """
    source: https://arrayfire.org/docs/group__image__func__moments.htm#ga3d0b4c037b137989f95131787882d9b4
    """
    out = ctypes.c_double(0)
    safe_call(_backend.clib.af_moments_all(ctypes.pointer(out), arr, moment.value))
    return out.value
