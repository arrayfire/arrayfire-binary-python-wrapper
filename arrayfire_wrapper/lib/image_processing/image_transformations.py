import ctypes

from arrayfire_wrapper.defines import AFArray, CDimT
from arrayfire_wrapper.lib._constants import Interp
from arrayfire_wrapper.lib._utility import call_from_clib


def resize(image: AFArray, odim0: int, odim1: int, method: Interp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__transform__func__resize.htm#gaa2fbcfb50e2b74acb0a087b17e4e0dd8
    """
    out = AFArray.create_null_pointer()
    call_from_clib(resize.__name__, ctypes.pointer(out), image, CDimT(odim0), CDimT(odim1), method.value)
    return out


def rotate(image: AFArray, theta: float, to_crop: bool, method: Interp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__transform__func__rotate.htm#ga826d224d5d5d0f721b7865a9d4c034da
    """
    out = AFArray.create_null_pointer()
    call_from_clib(resize.__name__, ctypes.pointer(out), image, ctypes.c_float(theta), to_crop, method.value)
    return out


def scale(image: AFArray, scale0: float, scale1: float, odim0: int, odim1: int, method: Interp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__transform__func__scale.htm#gafe489c0df047d7313e28955694f043d6
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        scale.__name__,
        ctypes.pointer(out),
        image,
        ctypes.c_float(scale0),
        ctypes.c_float(scale1),
        CDimT(odim0),
        CDimT(odim1),
        method.value,
    )
    return out


def skew(
    image: AFArray, skew0: float, skew1: float, odim0: int, odim1: int, method: Interp, is_inverse: bool, /
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__transform__func__skew.htm#ga469445c0345acad37cd0cfca2375c671
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        skew.__name__,
        ctypes.pointer(out),
        image,
        ctypes.c_float(skew0),
        ctypes.c_float(skew1),
        CDimT(odim0),
        CDimT(odim1),
        method.value,
        is_inverse,
    )
    return out


def transform(
    image: AFArray, transform_matrix: AFArray, odim0: int, odim1: int, method: Interp, is_inverse: bool, /
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__transform__func__transform.htm#ga43912cc2f13f3b08bb89e83e570ac346
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        transform.__name__,
        ctypes.pointer(out),
        image,
        transform_matrix,
        CDimT(odim0),
        CDimT(odim1),
        method.value,
        is_inverse,
    )
    return out


def transform_v2(
    image: AFArray, transform_matrix: AFArray, odim0: int, odim1: int, method: Interp, is_inverse: bool, /
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__transform__func__transform.htm#gac40849a4c191e999caae41cedeaa33bd
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        transform_v2.__name__,
        ctypes.pointer(out),
        image,
        transform_matrix,
        CDimT(odim0),
        CDimT(odim1),
        method.value,
        is_inverse,
    )
    return out


def transform_coordinates(image: AFArray, d0: float, d1: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__transform__func__coordinates.htm#gad6a3e6fde0cff195c8c308762e57bd52
    """
    out = AFArray.create_null_pointer()
    call_from_clib(transform_coordinates.__name__, ctypes.pointer(out), image, ctypes.c_float(d0), ctypes.c_float(d1))
    return out


def translate(image: AFArray, trans0: float, trans1: float, odim0: int, odim1: int, method: Interp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__transform__func__translate.htm#gaf753bd5ef4aabc810e68a93e38586aa5
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        translate.__name__,
        ctypes.pointer(out),
        image,
        ctypes.c_float(trans0),
        ctypes.c_float(trans1),
        CDimT(odim0),
        CDimT(odim1),
        method.value,
    )
    return out
