import ctypes
from typing import TYPE_CHECKING

from arrayfire_wrapper.backend import _backend
from arrayfire_wrapper.dtypes import CShape, Dtype

from ..._error_handler import safe_call

if TYPE_CHECKING:
    from arrayfire_wrapper._typing import AFArray


def constant(number: int | float, shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#gafc51b6a98765dd24cd4139f3bde00670
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_constant(
            ctypes.pointer(out), ctypes.c_double(number), 4, ctypes.pointer(c_shape.c_array), dtype.c_api_value
        )
    )
    return out


def constant_complex(number: int | float | complex, shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga5a083b1f3cd8a72a41f151de3bdea1a2
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_constant_complex(
            ctypes.pointer(out),
            ctypes.c_double(number.real),
            ctypes.c_double(number.imag),
            4,
            ctypes.pointer(c_shape.c_array),
            dtype.c_api_value,
        )
    )
    return out


def constant_long(number: int | float, shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga10f1c9fad1ce9e9fefd885d5a1d1fd49
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_constant_long(
            ctypes.pointer(out), ctypes.c_longlong(int(number.real)), 4, ctypes.pointer(c_shape.c_array)
        )
    )
    return out


def constant_ulong(number: int | float, shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga67af670cc9314589f8134019f5e68809
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_constant_ulong(
            ctypes.pointer(out), ctypes.c_ulonglong(int(number.real)), 4, ctypes.pointer(c_shape.c_array)
        )
    )
    return out
