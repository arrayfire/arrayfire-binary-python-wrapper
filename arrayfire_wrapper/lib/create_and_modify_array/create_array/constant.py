import ctypes

from arrayfire_wrapper.defines import AFArray, CShape
from arrayfire_wrapper.dtypes import Dtype
from arrayfire_wrapper.lib._utility import call_from_clib


def constant(number: int | float, shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#gafc51b6a98765dd24cd4139f3bde00670
    """
    out = AFArray.create_null_pointer()
    c_shape = CShape(*shape)

    call_from_clib(
        constant.__name__,
        ctypes.pointer(out),
        ctypes.c_double(number),
        4,
        ctypes.pointer(c_shape.c_array),
        dtype.c_api_value,
    )
    return out


def constant_complex(number: int | float | complex, shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga5a083b1f3cd8a72a41f151de3bdea1a2
    """
    out = AFArray.create_null_pointer()
    c_shape = CShape(*shape)

    call_from_clib(
        constant_complex.__name__,
        ctypes.pointer(out),
        ctypes.c_double(number.real),
        ctypes.c_double(number.imag),
        4,
        ctypes.pointer(c_shape.c_array),
        dtype.c_api_value,
    )
    return out


def constant_long(number: int | float, shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga10f1c9fad1ce9e9fefd885d5a1d1fd49
    """
    out = AFArray.create_null_pointer()
    c_shape = CShape(*shape)

    call_from_clib(
        constant_long.__name__,
        ctypes.pointer(out),
        ctypes.c_longlong(int(number.real)),
        4,
        ctypes.pointer(c_shape.c_array),
    )
    return out


def constant_ulong(number: int | float, shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga67af670cc9314589f8134019f5e68809
    """
    # out = ctypes.c_void_p(0)
    # out = AFArray(0)
    out = AFArray.create_null_pointer()
    c_shape = CShape(*shape)

    call_from_clib(
        constant_ulong.__name__,
        ctypes.pointer(out),
        ctypes.c_ulonglong(int(number.real)),
        4,
        ctypes.pointer(c_shape.c_array),
    )
    return out
