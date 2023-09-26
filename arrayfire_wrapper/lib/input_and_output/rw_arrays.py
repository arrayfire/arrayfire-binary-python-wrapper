import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def read_array_index(filename: str, idx: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__stream__func__read.htm#gab01a9d75d67f2ecfccac53b02c900930
    """
    out = AFArray.create_null_pointer()
    call_from_clib(read_array_index.__name__, ctypes.pointer(out), filename.encode("utf-8"), idx)
    return out


def read_array_key(filename: str, idx: str, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__stream__func__read.htm#ga8e0331b300f0b94ea9cc53606cf38278
    """
    out = AFArray.create_null_pointer()
    call_from_clib(read_array_key.__name__, ctypes.pointer(out), filename.encode("utf-8"), idx.encode("utf-8"))
    return out


def read_array_key_check(filename: str, idx: str, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__stream__func__read.htm#ga31522b71beee2b1c06d49b5aa65a5c6f
    """
    out = AFArray.create_null_pointer()
    call_from_clib(read_array_key_check.__name__, ctypes.pointer(out), filename.encode("utf-8"), idx.encode("utf-8"))
    return out


def save_array(key: str, arr: AFArray, filename: str, to_append: bool, /) -> int:
    """
    source: https://arrayfire.org/docs/group__stream__func__save.htm#ga3cb2f467e3f00b619cb2b4c727f1200b
    """
    out = ctypes.c_int(0)
    call_from_clib(
        save_array.__name__, ctypes.pointer(out), key.encode("utf-8"), arr, filename.encode("utf-8"), to_append
    )
    return out.value
