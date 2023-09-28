import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib
from arrayfire_wrapper.lib.features import AFFeatures


def gloh(
    arr: AFArray,
    n_layers: int,
    contrast_threshold: float,
    edge_threshold: float,
    init_sigma: float,
    double_input: bool,
    intensity_scale: float,
    feature_ratio: float,
    /,
) -> tuple[AFFeatures, AFArray]:
    """
    source: https://arrayfire.org/docs/group__cv__func__sift.htm#ga963dd3ecc8c17ff286a6caaee498b49c
    """

    out_features = AFFeatures.create_null_pointer()
    out_descriptors = AFArray.create_null_pointer()

    call_from_clib(
        gloh.__name__,
        ctypes.pointer(out_features),
        ctypes.pointer(out_descriptors),
        arr,
        ctypes.c_uint(n_layers),
        ctypes.c_float(contrast_threshold),
        ctypes.c_float(edge_threshold),
        ctypes.c_float(init_sigma),
        double_input,
        ctypes.c_float(intensity_scale),
        ctypes.c_float(feature_ratio),
    )
    return (out_features, out_descriptors)


def orb(
    arr: AFArray, fast_threshold: float, max_features: int, scale_factor: float, levels: int, blut_image: bool, /
) -> tuple[AFFeatures, AFArray]:
    """
    source: https://arrayfire.org/docs/group__cv__func__orb.htm#ga3b80df1c1f7d95ed1a52b73ba21d4d07
    """

    out_features = AFFeatures.create_null_pointer()
    out_descriptors = AFArray.create_null_pointer()

    call_from_clib(
        orb.__name__,
        ctypes.pointer(out_features),
        ctypes.pointer(out_descriptors),
        arr,
        ctypes.c_float(fast_threshold),
        ctypes.c_uint(max_features),
        ctypes.c_float(scale_factor),
        ctypes.c_uint(levels),
        blut_image,
    )
    return (out_features, out_descriptors)


def sift(
    arr: AFArray,
    n_layers: int,
    contrast_threshold: float,
    edge_threshold: float,
    init_sigma: float,
    double_input: bool,
    intensity_scale: float,
    feature_ratio: float,
    /,
) -> tuple[AFFeatures, AFArray]:
    """
    source: https://arrayfire.org/docs/group__cv__func__sift.htm#ga8bae580e5cd79b8adab7346146a5824f
    """

    out_features = AFFeatures.create_null_pointer()
    out_descriptors = AFArray.create_null_pointer()

    call_from_clib(
        sift.__name__,
        ctypes.pointer(out_features),
        ctypes.pointer(out_descriptors),
        arr,
        ctypes.c_uint(n_layers),
        ctypes.c_float(contrast_threshold),
        ctypes.c_float(edge_threshold),
        ctypes.c_float(init_sigma),
        double_input,
        ctypes.c_float(intensity_scale),
        ctypes.c_float(feature_ratio),
    )
    return (out_features, out_descriptors)
