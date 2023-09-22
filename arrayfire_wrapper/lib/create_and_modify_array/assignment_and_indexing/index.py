import ctypes
from typing import Any

from arrayfire_wrapper._backend import _backend
from arrayfire_wrapper.defines import AFArray, CDimT

from ..._error_handler import safe_call

# TODO unfinished module


def index_gen(arr: AFArray, ndims: int, indices: Any, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__index__func__index.htm#ga14a7d149dba0ed0b977335a3df9d91e6
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_index_gen(ctypes.pointer(out), arr, CDimT(ndims), indices.pointer))
    return out
