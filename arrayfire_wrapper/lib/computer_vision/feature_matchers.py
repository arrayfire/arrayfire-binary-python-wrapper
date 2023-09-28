import ctypes

from arrayfire_wrapper.defines import AFArray, CDimT
from arrayfire_wrapper.lib._constants import Match
from arrayfire_wrapper.lib._utility import call_from_clib


def hamming_matcher(query: AFArray, train: AFArray, dist_dim: int, n_dist: int, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__cv__func__hamming__matcher.htm#ga94ef1ed9b5214523725fff3f21af7a35
    """
    out_idx = AFArray.create_null_pointer()
    out_dist = AFArray.create_null_pointer()
    call_from_clib(
        hamming_matcher.__name__,
        ctypes.pointer(out_idx),
        ctypes.pointer(out_dist),
        query,
        train,
        CDimT(dist_dim),
        ctypes.c_uint(n_dist),
    )
    return (out_idx, out_dist)


def nearest_neighbour(
    query: AFArray, train: AFArray, dist_dim: int, n_dist: int, dist_type: Match, /
) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__cv__func__nearest__neighbour.htm#gaf46f2c5bf1ad2e71f94b58415837ffb2
    """
    out_idx = AFArray.create_null_pointer()
    out_dist = AFArray.create_null_pointer()
    call_from_clib(
        nearest_neighbour.__name__,
        ctypes.pointer(out_idx),
        ctypes.pointer(out_dist),
        query,
        train,
        CDimT(dist_dim),
        ctypes.c_uint(n_dist),
        dist_type.value,
    )
    return (out_idx, out_dist)
