import ctypes
from typing import Any

from arrayfire_wrapper.defines import AFArray, CDimT
from arrayfire_wrapper.lib._utility import call_from_clib

# TODO unfinished module


def index_gen(arr: AFArray, ndims: int, indices: Any, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__index__func__index.htm#ga14a7d149dba0ed0b977335a3df9d91e6
    """
    out = AFArray.create_null_pointer()
    call_from_clib(index_gen.__name__, ctypes.pointer(out), arr, CDimT(ndims), indices.pointer)
    return out
