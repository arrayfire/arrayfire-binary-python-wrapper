import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def set_intersect(first: AFArray, second: AFArray, is_unique: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__set__func__intersect.htm#ga985f9332c5f858eec66c717881ef2607
    """
    out = AFArray.create_null_pointer()
    call_from_clib(set_intersect.__name__, ctypes.pointer(out), first, second, ctypes.c_bool(is_unique))
    return out


def set_union(first: AFArray, second: AFArray, is_unique: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__set__func__union.htm#gaabeead0c0dc360db9398e9703dbb273f
    """
    out = AFArray.create_null_pointer()
    call_from_clib(set_union.__name__, ctypes.pointer(out), first, second, ctypes.c_bool(is_unique))
    return out


def set_unique(arr: AFArray, is_sorted: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__set__func__unique.htm#ga6afa1de48cbbc4b2df530c2530087943
    """
    out = AFArray.create_null_pointer()
    call_from_clib(set_unique.__name__, ctypes.pointer(out), arr, ctypes.c_bool(is_sorted))
    return out
