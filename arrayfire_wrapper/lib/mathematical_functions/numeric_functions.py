import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import binary_op, call_from_clib, unary_op
from arrayfire_wrapper.lib.mathematical_functions.arithmetic_operations import sub


def abs_(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__abs.htm#ga7e8b3c848e6cda3d1f3b0c8b2b4c3f8f
    """
    return unary_op(abs.__name__, arr)


def arg(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__arg.htm#gad04de0f7948688378dcd3628628a7424
    """
    return unary_op(arg.__name__, arr)


def ceil(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(ceil.__name__, arr)


def clamp(arr: AFArray, lo: AFArray, hi: AFArray, batch: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__clamp.htm#gac4e785c5c877c7905e56f44ef0cb5e61
    """
    out = AFArray.create_null_pointer()
    call_from_clib(clamp.__name__, ctypes.pointer(out), arr, lo, hi, ctypes.c_bool(batch))
    return out


def floor(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(floor.__name__, arr)


def hypot(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source:
    """
    return binary_op(hypot.__name__, lhs, rhs)


def maxof(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__max.htm#ga0cd47e70cf82b48730a97c59f494b421
    """
    return binary_op(maxof.__name__, lhs, rhs)


def minof(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__min.htm#ga2b842c2d86df978ff68699aeaafca794
    """
    return binary_op(minof.__name__, lhs, rhs)


def mod(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__mod.htm#ga01924d1b59d8886e46fabd2dc9b27e0f
    """
    return binary_op(mod.__name__, lhs, rhs)


def neg(arr: AFArray) -> AFArray:
    return sub(AFArray(0), arr)


def rem(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__clamp.htm#gac4e785c5c877c7905e56f44ef0cb5e61
    """
    return binary_op(rem.__name__, lhs, rhs)


def round_(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__sign.htm#ga2d55dfb9b25e0a1316b70f01d5b44b35
    """
    return unary_op(round.__name__, arr)


def sign(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__sign.htm#ga2d55dfb9b25e0a1316b70f01d5b44b35
    """
    return unary_op(sign.__name__, arr)


def trunc(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(trunc.__name__, arr)
