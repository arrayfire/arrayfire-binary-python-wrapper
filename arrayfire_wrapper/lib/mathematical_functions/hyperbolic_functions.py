from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import unary_op


def acosh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(acosh.__name__, arr)


def asinh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(asinh.__name__, arr)


def atanh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(atanh.__name__, arr)


def cosh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(cosh.__name__, arr)


def sinh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(sinh.__name__, arr)


def tanh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(tanh.__name__, arr)
