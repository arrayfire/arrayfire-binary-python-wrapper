import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._constants import BinaryOperator
from arrayfire_wrapper.lib._utility import call_from_clib


def accum(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__scan__func__accum.htm#ga50d499e844e0b63e338cb3ea50439629
    """
    out = AFArray.create_null_pointer()
    call_from_clib(accum.__name__, ctypes.pointer(out), arr, ctypes.c_int(dim))
    return out


def scan(arr: AFArray, dim: int, op: BinaryOperator, inclusive_scan: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__scan__func__scan.htm#ga1c864e22826f61bec2e9b6c61aa93fce
    """
    out = AFArray.create_null_pointer()
    call_from_clib(scan.__name__, ctypes.pointer(out), arr, dim, op.value, inclusive_scan)
    return out


def scan_by_key(key: AFArray, arr: AFArray, dim: int, op: BinaryOperator, inclusive_scan: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__scan__func__scanbykey.htm#gaaae150e0f197782782f45340d137b027
    """
    out = AFArray.create_null_pointer()
    call_from_clib(scan_by_key.__name__, ctypes.pointer(out), key, arr, dim, op.value, inclusive_scan)
    return out


def where(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__scan__func__where.htm#gafda59a3d25d35238592dd09907be9d07
    """
    out = AFArray.create_null_pointer()
    call_from_clib(where.__name__, ctypes.pointer(out), arr)
    return out
