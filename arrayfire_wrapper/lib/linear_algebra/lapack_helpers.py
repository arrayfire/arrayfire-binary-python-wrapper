import ctypes

from arrayfire_wrapper.lib._utility import call_from_clib


def is_lapack_available() -> bool:
    """
    source: https://arrayfire.org/docs/group__lapack__helper__func__available.htm#gaf96dab15121b5cf3599fa9dbc7257f33
    """
    out = ctypes.c_bool(False)
    call_from_clib(is_lapack_available.__name__, ctypes.pointer(out))
    return bool(out.value)
