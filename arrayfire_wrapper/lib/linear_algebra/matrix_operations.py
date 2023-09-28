import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._constants import MatProp, Norm
from arrayfire_wrapper.lib._utility import call_from_clib


def det(arr: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__lapack__ops__func__det.htm#gad9ec24f92c8312f4c6c5eb2536f1618e
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(det.__name__, ctypes.pointer(real), ctypes.pointer(imag), arr)
    return real.value if imag == 0 else real.value + imag.value * 1j


def inverse(arr: AFArray, options: MatProp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__lapack__ops__func__inv.htm#ga7a059e2de445dc788aa46ecd1e265d78
    """
    out = AFArray.create_null_pointer()
    call_from_clib(inverse.__name__, ctypes.pointer(out), arr, options.value)
    return out


def norm(arr: AFArray, norm_type: Norm, p: float, q: float, /) -> float:
    """
    source: https://arrayfire.org/docs/group__lapack__ops__func__norm.htm#ga5bee140b7afdb1a300960e3004bf624c
    """
    out = ctypes.c_double(0)
    call_from_clib(norm.__name__, ctypes.pointer(out), arr, norm_type.value, ctypes.c_double(p), ctypes.c_double(q))
    return out.value


def pinverse(arr: AFArray, tol: float, options: MatProp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__lapack__ops__func__pinv.htm#ga00adc3678f4829c40d50359638f61c0c
    """
    out = AFArray.create_null_pointer()
    call_from_clib(pinverse.__name__, ctypes.pointer(out), arr, ctypes.c_double(tol), options.value)
    return out


def rank(arr: AFArray, tol: float, /) -> int:
    """
    source: https://arrayfire.org/docs/group__lapack__ops__func__rank.htm#ga787b979a7e3f1e8aec4a8ec2d96a1940
    """
    out = ctypes.c_uint(0)
    call_from_clib(rank.__name__, ctypes.pointer(out), arr, ctypes.c_double(tol))
    return out.value
