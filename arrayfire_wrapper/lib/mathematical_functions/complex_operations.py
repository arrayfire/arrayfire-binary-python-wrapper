from arrayfire_wrapper._backend import _backend
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import binary_op, unary_op


def cplx(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_cplx, arr)


def cplx2(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source:
    """
    return binary_op(_backend.clib.af_cplx2, lhs, rhs)


def conjg(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_conjg, arr)


def imag(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_imag, arr)


def real(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_real, arr)
