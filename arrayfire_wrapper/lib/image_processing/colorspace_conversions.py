import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._constants import CSpace, YCCStd
from arrayfire_wrapper.lib._utility import call_from_clib


def color_space(image: AFArray, to_type: CSpace, from_type: CSpace, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__colorspace.htm#ga6d003d4193491aaa80a386fce711a24d
    """
    out = AFArray.create_null_pointer()
    call_from_clib(color_space.__name__, ctypes.pointer(out), image, to_type.value, from_type.value)
    return out


def gray2rgb(image: AFArray, r_factor: float, g_factor: float, b_factor: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__gray2rgb.htm#gaa966ed1b88470cc8bdba1128c445e4c0
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        gray2rgb.__name__,
        ctypes.pointer(out),
        image,
        ctypes.c_float(r_factor),
        ctypes.c_float(g_factor),
        ctypes.c_float(b_factor),
    )
    return out


def hsv2rgb(image: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__hsv2rgb.htm#ga3191a0eb0c0a3aceb981c5dab4b8eefb
    """
    out = AFArray.create_null_pointer()
    call_from_clib(hsv2rgb.__name__, ctypes.pointer(out), image)
    return out


def rgb2gray(image: AFArray, r_factor: float, g_factor: float, b_factor: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__rgb2gray.htm#ga6cc36e0f3fc7a291a759b047622b2d56
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        rgb2gray.__name__,
        ctypes.pointer(out),
        image,
        ctypes.c_float(r_factor),
        ctypes.c_float(g_factor),
        ctypes.c_float(b_factor),
    )
    return out


def rgb2hsv(image: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__rgb2hsv.htm#ga9fd64ad534fb69185af01c53a0aacceb
    """
    out = AFArray.create_null_pointer()
    call_from_clib(rgb2hsv.__name__, ctypes.pointer(out), image)
    return out


def rgb2ycbcr(image: AFArray, standard: YCCStd, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__rgb2ycbcr.htm#ga808a8f0f280fe4330828ad0428a57b9e
    """
    out = AFArray.create_null_pointer()
    call_from_clib(rgb2ycbcr.__name__, ctypes.pointer(out), image, standard.value)
    return out


def ycbcr2rgb(image: AFArray, standard: YCCStd, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__ycbcr2rgb.htm#ga94217a2f6cda2e867d134eca5dbca2b0
    """
    out = AFArray.create_null_pointer()
    call_from_clib(rgb2ycbcr.__name__, ctypes.pointer(out), image, standard.value)
    return out
