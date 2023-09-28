from __future__ import annotations

import ctypes

from arrayfire_wrapper.defines import AFArray, CDimT, _AFBase
from arrayfire_wrapper.lib._utility import call_from_clib


class AFFeatures(_AFBase):
    """
    source: https://arrayfire.org/docs/features_8h.htm#a294c8f0e20b10dfc4f4f18566dba06bc
    """

    pass


def create_features(num: int, /) -> AFFeatures:
    """
    source: https://arrayfire.org/docs/group__features__group__features.htm#gac3ffa28d0fcefabf4d686934a2fbd106
    """
    out = AFFeatures.create_null_pointer()
    call_from_clib(create_features.__name__, ctypes.pointer(out), CDimT(num))
    return out


def get_features_num(features: AFFeatures, /) -> int:
    """
    source: https://arrayfire.org/docs/group__features__group__features.htm#ga9d5a64f8c3a3a3033e82592b1e2e5980
    """
    out = CDimT(0)
    call_from_clib(get_features_num.__name__, ctypes.pointer(out), features)
    return out.value


def get_features_orientation(features: AFFeatures, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__features__group__features.htm#ga68f36385a0cbee21c068b691d6dcece9
    """
    out = AFArray.create_null_pointer()
    call_from_clib(get_features_orientation.__name__, ctypes.pointer(out), features)
    return out


def get_features_score(features: AFFeatures, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__features__group__features.htm#ga80fd97bd5c4506aba503890ff29fdaa7
    """
    out = AFArray.create_null_pointer()
    call_from_clib(get_features_score.__name__, ctypes.pointer(out), features)
    return out


def get_features_size(features: AFFeatures, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__features__group__features.htm#ga2588cc6b941053a8476ef5b6eba5cd0a
    """
    out = AFArray.create_null_pointer()
    call_from_clib(get_features_size.__name__, ctypes.pointer(out), features)
    return out


def get_features_xpos(features: AFFeatures, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__features__group__features.htm#gaf2fae4f54b124011e1ea597630a07fc0
    """
    out = AFArray.create_null_pointer()
    call_from_clib(get_features_xpos.__name__, ctypes.pointer(out), features)
    return out


def get_features_ypos(features: AFFeatures, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__features__group__features.htm#ga1dea28b2b3e623b10ab8a6e3a0314424
    """
    out = AFArray.create_null_pointer()
    call_from_clib(get_features_ypos.__name__, ctypes.pointer(out), features)
    return out


def release_features(features: AFFeatures, /) -> None:
    """
    source: https://arrayfire.org/docs/group__features__group__features.htm#ga84f6cb7e56185ceaaeb5472e54ba1889
    """
    call_from_clib(release_features.__name__, features)
    return None


def retain_features(features: AFFeatures, /) -> AFFeatures:
    """
    source: https://arrayfire.org/docs/group__features__group__features.htm#ga4e44ce81fa53fbe4e19f786e450255ae
    """
    out = AFFeatures.create_null_pointer()
    call_from_clib(get_features_ypos.__name__, ctypes.pointer(out), features)
    return out
