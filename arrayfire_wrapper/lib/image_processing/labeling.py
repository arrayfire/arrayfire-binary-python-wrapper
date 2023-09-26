import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.dtypes import Dtype
from arrayfire_wrapper.lib._constants import Connectivity
from arrayfire_wrapper.lib._utility import call_from_clib


def confidence_cc(
    image: AFArray, seed_x: AFArray, seed_y: AFArray, radius: int, multiplier: int, iter: int, segmented_value: float
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__confidence__cc.htm#gaace9e5c33fc46076177e546a5642095c
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        confidence_cc.__name__,
        ctypes.pointer(out),
        image,
        seed_x,
        seed_y,
        ctypes.c_uint(radius),
        ctypes.c_uint(multiplier),
        ctypes.c_int(iter),
        ctypes.c_double(segmented_value),
    )
    return out


def regions(image: AFArray, connectivity: Connectivity, dtype: Dtype) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__regions.htm#ga936ee980be71bf26cd3d238d3db1ed34
    """
    out = AFArray.create_null_pointer()
    call_from_clib(regions.__name__, ctypes.pointer(out), image, connectivity.value, dtype.c_api_value)
    return out
