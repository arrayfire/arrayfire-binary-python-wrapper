import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def hist_equal(image: AFArray, hist: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__histequal.htm#ga2ef6316afcfde69eff2b71c41602f347
    """
    out = AFArray.create_null_pointer()
    call_from_clib(hist_equal.__name__, ctypes.pointer(out), image, hist)
    return out


def histogram(image: AFArray, nbins: int, min_val: float, max_val: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__histogram.htm#gab14404cc8923cf3e1bd0d9d94ef63325
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        histogram.__name__,
        ctypes.pointer(out),
        ctypes.c_uint(nbins),
        ctypes.c_double(min_val),
        ctypes.c_double(max_val),
    )
    return out
