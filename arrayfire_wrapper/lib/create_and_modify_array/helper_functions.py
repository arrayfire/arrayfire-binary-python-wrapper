import ctypes

from arrayfire_wrapper._backend import _backend
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.dtypes import Dtype, to_str
from arrayfire_wrapper.lib._error_handler import safe_call
from arrayfire_wrapper.lib._utility import unary_op


def cast(arr: AFArray, dtype: Dtype) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__cast.htm#gab0cb307d6f9019ac8cbbbe9b8a4d6b9b
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_cast(ctypes.pointer(out), arr, dtype.c_api_value))
    return out


def isinf(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__isinf.htm#ga933758a10227f15697ff503339e86823
    """
    return unary_op(_backend.clib.af_isinf, arr)


def isnan(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__isnan.htm#ga40a48fc1cd94ff02f6ddeb7dafd1f87e
    """
    return unary_op(_backend.clib.af_isnan, arr)


def iszero(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__iszero.htm#ga559003777ce5148277b07903c351ecea
    """
    return unary_op(_backend.clib.af_iszero, arr)


def array_to_string(exp: str, arr: AFArray, precision: int, transpose: bool, /) -> str:
    """
    source: https://arrayfire.org/docs/group__print__func__tostring.htm#ga01f32ef2420b5d4592c6e4b4964b863b
    """
    out = ctypes.c_char_p(0)
    safe_call(_backend.clib.af_array_to_string(ctypes.pointer(out), exp, arr, precision, transpose))
    return to_str(out)
