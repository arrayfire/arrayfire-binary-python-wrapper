import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib
from arrayfire_wrapper.lib.features import AFFeatures


def dog(arr: AFArray, radius1: int, radius2: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__cv__func__dog.htm#ga6621c294343a7ff8aa5e1f6a5b2eca00
    """
    out = AFArray.create_null_pointer()
    call_from_clib(dog.__name__, ctypes.pointer(out), arr, radius1, radius2)
    return out


def fast(
    arr: AFArray, threshold: float, arc_length: int, non_max: bool, feature_ratio: float, edge: int
) -> AFFeatures:
    """
    source: https://arrayfire.org/docs/group__cv__func__fast.htm#ga74872a7c8aa4f57541dc8bf6ddc7447e
    """
    out = AFFeatures.create_null_pointer()
    call_from_clib(
        fast.__name__,
        ctypes.pointer(out),
        arr,
        ctypes.c_float(threshold),
        ctypes.c_uint(arc_length),
        non_max,
        ctypes.c_float(feature_ratio),
        ctypes.c_uint(edge),
    )
    return out


def harris(
    arr: AFArray, max_corners: int, min_response: float, sigma: float, block_size: int, k_threshold: float
) -> AFFeatures:
    """
    source: https://arrayfire.org/docs/group__cv__func__harris.htm#ga5dbef6bcdba838236b67e5d287ae42fc
    """
    out = AFFeatures.create_null_pointer()
    call_from_clib(
        harris.__name__,
        arr,
        ctypes.c_uint(max_corners),
        ctypes.c_float(min_response),
        ctypes.c_float(sigma),
        ctypes.c_uint(block_size),
        ctypes.c_float(k_threshold),
    )
    return out


def susan(
    arr: AFArray, radius: int, diff_threshold: float, geom_threshold: float, feature_ratio: float, edge: int
) -> AFFeatures:
    """
    source: https://arrayfire.org/docs/group__cv__func__susan.htm#ga4c38aa6f12bce96e7198b28f9479d99f
    """
    out = AFFeatures.create_null_pointer()
    call_from_clib(
        susan.__name__,
        ctypes.pointer(out),
        arr,
        ctypes.c_uint(radius),
        ctypes.c_float(diff_threshold),
        ctypes.c_float(geom_threshold),
        ctypes.c_float(feature_ratio),
        ctypes.c_uint(edge),
    )
    return out
