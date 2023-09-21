import ctypes

from arrayfire_wrapper._backend import _backend
from arrayfire_wrapper.defines import AFArray, ArrayBuffer, CDimT, CShape, CType, PointerSource
from arrayfire_wrapper.dtypes import Dtype
from arrayfire_wrapper.lib._error_handler import safe_call


def create_strided_array(
    shape: tuple[int, ...],
    dtype: Dtype,
    array_buffer: ArrayBuffer,
    offset: CType,
    strides: tuple[int, ...],
    pointer_source: PointerSource,
    /,
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__internal__func__create.htm#gad31241a3437b7b8bc3cf49f85e5c4e0c
    """
    out = AFArray.create_null_pointer()
    c_shape = CShape(*shape)

    if offset is None:
        offset = CDimT(0)

    if strides is None:
        strides = (1, c_shape[0], c_shape[0] * c_shape[1], c_shape[0] * c_shape[1] * c_shape[2])

    if len(strides) < 4:
        strides += (strides[-1],) * (4 - len(strides))

    safe_call(
        _backend.clib.af_create_strided_array(
            ctypes.pointer(out),
            ctypes.c_void_p(array_buffer.address),
            offset,
            c_shape.original_shape,
            ctypes.pointer(c_shape.c_array),
            CShape(*strides).c_array,
            dtype.c_api_value,
            pointer_source.value,
        )
    )
    return out


def get_offset(arr: AFArray, /) -> int:
    """
    source: https://arrayfire.org/docs/group__internal__func__offset.htm#ga303cb334026bdb5cab86e038951d6a5a
    """
    out = CDimT(0)
    safe_call(_backend.clib.af_get_offset(ctypes.pointer(out), arr))
    return out.value


def get_strides(arr: AFArray) -> tuple[int, ...]:
    """
    source: https://arrayfire.org/docs/group__internal__func__strides.htm#gaff91b376156ce0ad7180af6e68faab51
    """
    s0 = CDimT(0)
    s1 = CDimT(0)
    s2 = CDimT(0)
    s3 = CDimT(0)
    safe_call(
        _backend.clib.af_get_strides(
            ctypes.pointer(s0), ctypes.pointer(s1), ctypes.pointer(s2), ctypes.pointer(s3), arr
        )
    )
    return (s0.value, s1.value, s2.value, s3.value)


def is_linear(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__internal__func__linear.htm#gaef70fb8591d16522a5b05bd7467e7d8a
    """
    out = ctypes.c_bool(False)
    safe_call(_backend.clib.af_is_linear(ctypes.pointer(out), arr))
    return bool(out.value)


def is_owner(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__internal__func__owner.htm#ga9db0eda6ffa4c49e7f66c3d53d62186a
    """
    out = ctypes.c_bool(False)
    safe_call(_backend.clib.af_is_owner(ctypes.pointer(out), arr))
    return bool(out.value)
