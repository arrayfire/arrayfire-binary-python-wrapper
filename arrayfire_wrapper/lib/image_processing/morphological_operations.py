import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def dilate(image: AFArray, mask: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__dilate.htm#ga42a4df68ce13f8d9b1bc50f56a5f0927
    """
    out = AFArray.create_null_pointer()
    call_from_clib(dilate.__name__, ctypes.pointer(out), image, mask)
    return out


def dilate3(image: AFArray, mask: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__dilate3d.htm#gaf87ec9878f8405f18324a8e532cd7f2e
    """
    out = AFArray.create_null_pointer()
    call_from_clib(dilate3.__name__, ctypes.pointer(out), image, mask)
    return out


def erode(image: AFArray, mask: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__erode.htm#ga6420bd50feac6ca3426ebf2509c8431e
    """
    out = AFArray.create_null_pointer()
    call_from_clib(erode.__name__, ctypes.pointer(out), image, mask)
    return out


def erode3(image: AFArray, mask: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__erode3d.htm#ga94b15e5a5721c5f5daa511d0a830e26a
    """
    out = AFArray.create_null_pointer()
    call_from_clib(erode3.__name__, ctypes.pointer(out), image, mask)
    return out
