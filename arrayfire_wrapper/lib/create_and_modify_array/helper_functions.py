import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.dtypes import Dtype, to_str
from arrayfire_wrapper.lib._utility import call_from_clib, unary_op


def cast(arr: AFArray, dtype: Dtype) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__cast.htm#gab0cb307d6f9019ac8cbbbe9b8a4d6b9b
    """
    out = AFArray.create_null_pointer()
    call_from_clib(cast.__name__, ctypes.pointer(out), arr, dtype.c_api_value)
    return out


def isinf(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__isinf.htm#ga933758a10227f15697ff503339e86823
    """
    return unary_op(isinf.__name__, arr)


def isnan(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__isnan.htm#ga40a48fc1cd94ff02f6ddeb7dafd1f87e
    """
    return unary_op(isnan.__name__, arr)


def iszero(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__iszero.htm#ga559003777ce5148277b07903c351ecea
    """
    return unary_op(iszero.__name__, arr)


def array_to_string(exp: str, arr: AFArray, precision: int, transpose: bool, /) -> str:
    """
    source: https://arrayfire.org/docs/group__print__func__tostring.htm#ga01f32ef2420b5d4592c6e4b4964b863b
    """
    out = ctypes.c_char_p(0)
    call_from_clib(array_to_string.__name__, ctypes.pointer(out), exp, arr, precision, transpose)
    return to_str(out)
