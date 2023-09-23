from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import binary_op, unary_op


def cplx(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(cplx.__name__, arr)


def cplx2(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source:
    """
    return binary_op(cplx2.__name__, lhs, rhs)


def conjg(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(conjg.__name__, arr)


def imag(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(imag.__name__, arr)


def real(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(real.__name__, arr)
