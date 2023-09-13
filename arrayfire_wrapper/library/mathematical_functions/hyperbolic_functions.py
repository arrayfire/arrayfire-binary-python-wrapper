from typing import TYPE_CHECKING

from arrayfire_wrapper.backend import _backend
from arrayfire_wrapper.library._utility import unary_op

if TYPE_CHECKING:
    from arrayfire_wrapper._typing import AFArray


def acosh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_acosh, arr)


def asinh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_asinh, arr)


def atanh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_atanh, arr)


def cosh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_cosh, arr)


def sinh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_sinh, arr)


def tanh(arr: AFArray, /) -> AFArray:
    """
    source:
    """
    return unary_op(_backend.clib.af_tanh, arr)
