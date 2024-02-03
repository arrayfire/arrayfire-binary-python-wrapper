import ctypes

from arrayfire_wrapper.defines import AFArray, CDimT
from arrayfire_wrapper.lib._constants import CannyThreshold, Diffusion, Flux, IterativeDeconv, Pad
from arrayfire_wrapper.lib._utility import call_from_clib


def anisotropic_diffusion(
    image: AFArray, timestep: float, conductance: float, iterations: int, fftype: Flux, diffusion_kind: Diffusion, /
) -> AFArray:
    """
    source:https://arrayfire.org/docs/group__image__func__anisotropic__diffusion.htm#ga4ef489c34a5dbc29bf9ec28c4d8aa310
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        anisotropic_diffusion.__name__,
        ctypes.pointer(out),
        image,
        ctypes.c_float(timestep),
        ctypes.c_float(conductance),
        ctypes.c_uint(iterations),
        fftype.value,
        diffusion_kind.value,
    )
    return out


def bilateral(image: AFArray, spatial_sigma: float, chromatic_sigma: float, is_color: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__bilateral.htm#gaa3af44395a7cfed60b01b2ea9bf06fdc
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        bilateral.__name__,
        ctypes.pointer(out),
        image,
        ctypes.c_float(spatial_sigma),
        ctypes.c_float(chromatic_sigma),
        is_color,
    )
    return out


def canny(
    image: AFArray,
    threshold_type: CannyThreshold,
    low_threshold_ratio: float,
    high_threshold_ratio: float,
    sobel_window: int,
    is_fast: bool,
    /,
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__canny.htm#ga3cc80f80d1b2a19312d033763b382957
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        canny.__name__,
        ctypes.pointer(out),
        image,
        threshold_type.value,
        ctypes.c_float(low_threshold_ratio),
        ctypes.c_float(high_threshold_ratio),
        ctypes.c_uint(sobel_window),
        ctypes.c_bool(is_fast),
    )
    return out


def inverse_deconv(image: AFArray, psf: AFArray, gamma: float, algo: IterativeDeconv) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__inverse__deconv.htm#ga4b7c66c24db219c1058890ec6beeef7c
    """
    out = AFArray.create_null_pointer()
    call_from_clib(inverse_deconv.__name__, ctypes.pointer(out), image, psf, ctypes.c_float(gamma), algo.value)
    return out


def iterative_deconv(
    image: AFArray, psf: AFArray, iterations: int, relax_factor: float, algo: IterativeDeconv
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__iterative__deconv.htm#gaf716e895be65781a1d046619409716b7
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        iterative_deconv.__name__,
        ctypes.pointer(out),
        image,
        psf,
        ctypes.c_uint(iterations),
        ctypes.c_float(relax_factor),
        algo.value,
    )
    return out


def maxfilt(image: AFArray, wind_length: int, wind_width: int, edge_pad: Pad, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__maxfilt.htm#ga97e07bf5f5c58752d23d1772586b71f4
    """
    out = AFArray.create_null_pointer()
    call_from_clib(maxfilt.__name__, ctypes.pointer(out), image, CDimT(wind_length), CDimT(wind_width), edge_pad.value)
    return out


def mean_shift(image: AFArray, spatial_sigma: float, chromatic_sigma: float, iter: int, is_color: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__mean__shift.htm#ga3d39d9d7563f2cf6c94892e1f86e2b5b
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        mean_shift.__name__,
        ctypes.pointer(out),
        ctypes.c_float(spatial_sigma),
        ctypes.c_float(chromatic_sigma),
        ctypes.c_uint(iter),
        is_color,
    )
    return out


def medfilt(image: AFArray, wind_length: int, wind_width: int, edge_pad: Pad, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__medfilt.htm#gaaf3f62f2de0f4dc315b831e494e1b2c0
    """
    out = AFArray.create_null_pointer()
    call_from_clib(medfilt.__name__, ctypes.pointer(out), image, CDimT(wind_length), CDimT(wind_width), edge_pad.value)
    return out


def medfilt1(image: AFArray, wind_length: int, edge_pad: Pad, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__medfilt.htm#gad108ea62cbbb5371bd14a17d06384359
    """
    out = AFArray.create_null_pointer()
    call_from_clib(medfilt1.__name__, ctypes.pointer(out), image, CDimT(wind_length), edge_pad.value)
    return out


def medfilt2(image: AFArray, wind_length: int, wind_width: int, edge_pad: Pad, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__medfilt.htm#ga2cb99dca5842f74f6b9cd28eb187a9cd
    """
    out = AFArray.create_null_pointer()
    call_from_clib(medfilt.__name__, ctypes.pointer(out), image, CDimT(wind_length), CDimT(wind_width), edge_pad.value)
    return out


def minfilt(image: AFArray, wind_length: int, wind_width: int, edge_pad: Pad, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__minfilt.htm#ga39ce8056585b157af1c854f1b5f3e23a
    """
    out = AFArray.create_null_pointer()
    call_from_clib(minfilt.__name__, ctypes.pointer(out), image, CDimT(wind_length), CDimT(wind_width), edge_pad.value)
    return out


def sat(image: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__sat.htm#ga8ac0fccad8bf0105839ee734a7796407
    """
    out = AFArray.create_null_pointer()
    call_from_clib(sat.__name__, ctypes.pointer(out), image)
    return out


def sobel_operator(image: AFArray, kernel_size: int, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__image__func__sobel.htm#ga40e382358b5c1c16318cc8bf094eaf4c
    """
    out_dx = AFArray.create_null_pointer()
    out_dy = AFArray.create_null_pointer()
    call_from_clib(
        sobel_operator.__name__, ctypes.pointer(out_dx), ctypes.pointer(out_dy), image, ctypes.c_uint(kernel_size)
    )
    return (out_dx, out_dy)
