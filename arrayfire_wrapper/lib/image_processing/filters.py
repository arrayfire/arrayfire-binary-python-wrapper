import ctypes

from arrayfire_wrapper._backend import _backend
from arrayfire_wrapper.defines import AFArray, CDimT
from arrayfire_wrapper.lib._constants import Pad
from arrayfire_wrapper.lib._error_handler import safe_call


def maxfilt(arr: AFArray, wind_lenght: int, wind_width: int, edge_pad: Pad) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__maxfilt.htm#ga97e07bf5f5c58752d23d1772586b71f4
    """
    out = AFArray.create_null_pointer()
    safe_call(
        _backend.clib.af_maxfilt(ctypes.pointer(out), arr, CDimT(wind_lenght), CDimT(wind_width), edge_pad.value)
    )
    return out
