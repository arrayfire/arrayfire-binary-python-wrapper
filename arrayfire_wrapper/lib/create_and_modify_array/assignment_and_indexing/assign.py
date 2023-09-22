import ctypes
from typing import Any

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib

# TODO fix typing for indices across all functions


def assign_gen(lhs: AFArray, rhs: AFArray, ndims: int, indices: Any, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__index__func__assign.htm#ga93cd5199c647dce0e3b823f063b352ae
    """
    out = AFArray(0)
    call_from_clib(assign_gen.__name__, ctypes.pointer(out), lhs, ndims, indices.pointer, rhs)
    return out


def assign_seq(lhs: AFArray, rhs: AFArray, ndims: int, indices: Any, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__index__func__assign.htm#ga3b201c3114941b6f8d0e344afcd18457
    """
    out = AFArray(0)
    call_from_clib(assign_seq.__name__, ctypes.pointer(out), lhs, ndims, indices.pointer, rhs)
    return out
