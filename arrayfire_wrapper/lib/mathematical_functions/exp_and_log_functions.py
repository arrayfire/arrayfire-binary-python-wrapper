from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import binary_op, unary_op


def cbrt(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(cbrt.__name__, arr)


def erf(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(erf.__name__, arr)


def erfc(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(erfc.__name__, arr)


def exp(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(exp.__name__, arr)


def expm1(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(expm1.__name__, arr)


def factorial(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(factorial.__name__, arr)


def lgamma(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(lgamma.__name__, arr)


def log(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(log.__name__, arr)


def log10(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(log10.__name__, arr)


def log1p(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(log1p.__name__, arr)


def log2(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(log2.__name__, arr)


def pow(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__pow.htm#ga0f28be1a9c8b176a78c4a47f483e7fc6
    """
    return binary_op(pow.__name__, lhs, rhs)


def pow2(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(pow2.__name__, arr)


def root(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source:
    """
    return binary_op(root.__name__, lhs, rhs)


def rsqrt(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(rsqrt.__name__, arr)


def sqrt(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(sqrt.__name__, arr)


def sigmoid(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__sigmoid.htm#gadf4280e3283b65264de75194e0a6d565
    """
    return unary_op(sigmoid.__name__, arr)


def tgamma(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(tgamma.__name__, arr)
