from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import binary_op, unary_op


def acos(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(acos.__name__, arr)


def asin(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(asin.__name__, arr)


def atan(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(atan.__name__, arr)


def atan2(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source:
    """
    return binary_op(atan2.__name__, lhs, rhs)


def cos(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(cos.__name__, arr)


def sin(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(sin.__name__, arr)


def tan(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(tan.__name__, arr)
