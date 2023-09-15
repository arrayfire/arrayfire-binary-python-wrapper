from arrayfire_wrapper.backend import _backend
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.library._utility import binary_op, unary_op


def acos(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_acos, arr)


def asin(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_asin, arr)


def atan(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_atan, arr)


def atan2(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source:
    """
    return binary_op(_backend.clib.af_atan2, lhs, rhs)


def cos(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_cos, arr)


def sin(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_sin, arr)


def tan(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_tan, arr)
