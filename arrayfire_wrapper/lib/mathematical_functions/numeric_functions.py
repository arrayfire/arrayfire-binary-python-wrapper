import ctypes

from arrayfire_wrapper._backend import _backend
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import binary_op, unary_op

from .._error_handler import safe_call


def abs_(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__abs.htm#ga7e8b3c848e6cda3d1f3b0c8b2b4c3f8f
    """
    return unary_op(_backend.clib.af_abs, arr)


def arg(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__arg.htm#gad04de0f7948688378dcd3628628a7424
    """
    return unary_op(_backend.clib.af_arg, arr)


def ceil(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_ceil, arr)


def clamp(arr: AFArray, /, lo: float, hi: float) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__clamp.htm#gac4e785c5c877c7905e56f44ef0cb5e61
    """
    # TODO: check if lo and hi are of type float. Can be ArrayFire array as well
    out = AFArray(0)
    safe_call(_backend.clib.af_clamp(ctypes.pointer(out), arr, lo, hi))
    return out


def floor(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_floor, arr)


def hypot(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source:
    """
    return binary_op(_backend.clib.af_hypot, lhs, rhs)


def max_(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__max.htm#ga0cd47e70cf82b48730a97c59f494b421
    """
    return binary_op(_backend.clib.af_maxof, lhs, rhs)


def min_(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__min.htm#ga2b842c2d86df978ff68699aeaafca794
    """
    return binary_op(_backend.clib.af_minof, lhs, rhs)


def mod(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__mod.htm#ga01924d1b59d8886e46fabd2dc9b27e0f
    """
    return binary_op(_backend.clib.af_mod, lhs, rhs)


def neg(arr: AFArray) -> AFArray:
    # TODO
    return NotImplemented


def rem(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__clamp.htm#gac4e785c5c877c7905e56f44ef0cb5e61
    """
    return binary_op(_backend.clib.af_rem, lhs, rhs)


def round_(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__sign.htm#ga2d55dfb9b25e0a1316b70f01d5b44b35
    """
    return unary_op(_backend.clib.af_round, arr)


def sign(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__sign.htm#ga2d55dfb9b25e0a1316b70f01d5b44b35
    """
    return unary_op(_backend.clib.af_sign, arr)


def trunc(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_trunc, arr)
