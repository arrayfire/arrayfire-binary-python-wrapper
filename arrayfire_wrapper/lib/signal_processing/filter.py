import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def fir(b: AFArray, x: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fir.htm#ga3b9636788162beebb313fa8cc67ac8a7
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fir.__name__, ctypes.pointer(out), b, x)
    return out


def iir(b: AFArray, a: AFArray, x: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__iir.htm#ga2ccd475bfe8c3ca5df87639595d12d68
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fir.__name__, ctypes.pointer(out), b, a, x)
    return out
