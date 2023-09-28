import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def gaussian_kernel(rows: int, cols: int, sigma_r: float, sigma_c: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__gauss.htm#gaae27509f852c88f97fb31df2d5bdef2b
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        gaussian_kernel.__name__,
        ctypes.pointer(out),
        ctypes.c_int(rows),
        ctypes.c_int(cols),
        ctypes.c_double(sigma_r),
        ctypes.c_double(sigma_c),
    )
    return out
