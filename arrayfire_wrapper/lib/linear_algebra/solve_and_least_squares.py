import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._constants import MatProp
from arrayfire_wrapper.lib._utility import call_from_clib


def solve(a: AFArray, b: AFArray, options: MatProp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__lapack__solve__func__gen.htm#gae0312f53baae527a64e2593a8cc744eb
    """
    out = AFArray.create_null_pointer()
    call_from_clib(solve.__name__, ctypes.pointer(out), a, b, options.value)
    return out


def solve_lu(a: AFArray, b: AFArray, pivot: AFArray, options: MatProp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__lapack__solve__lu__func__gen.htm#gaf67856876581d98e4c3b7a6f0cfa7c88
    """
    out = AFArray.create_null_pointer()
    call_from_clib(solve_lu.__name__, ctypes.pointer(out), a, pivot, b, options.value)
    return out
