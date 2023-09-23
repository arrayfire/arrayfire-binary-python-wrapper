import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def diff1(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__calc__func__diff1.htm#gad3be33ce8114f65c188645e958fce171
    """
    out = AFArray.create_null_pointer()
    call_from_clib(diff1.__name__, ctypes.pointer(out), arr, ctypes.c_int(dim))
    return out


def diff2(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__calc__func__diff2.htm#gafc7b2d05e4e85aeb3e8b3239f598f70c
    """
    out = AFArray.create_null_pointer()
    call_from_clib(diff2.__name__, ctypes.pointer(out), arr, ctypes.c_int(dim))
    return out


def gradient(arr: AFArray, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__calc__func__grad.htm#gadb342e6765c1536125261b035f7eee59
    """
    out_dx = AFArray.create_null_pointer()
    out_dy = AFArray.create_null_pointer()
    call_from_clib(gradient.__name__, ctypes.pointer(out_dx), ctypes.pointer(out_dy), arr)
    return (out_dx, out_dy)
