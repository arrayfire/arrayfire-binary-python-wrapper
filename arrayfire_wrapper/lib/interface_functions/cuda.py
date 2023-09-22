import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def cublas_set_math_mode(mode: int) -> None:
    """
    source: https://arrayfire.org/docs/group__cuda__mat.htm#gac23ea38f0bff77a0e12555f27f47aa4f
    """
    call_from_clib("cublasSetMathMode", mode, clib_prefix="afcu")
    return None


def get_native_id(index: int) -> int:
    """
    source: https://arrayfire.org/docs/group__cuda__mat.htm#gaf38af1cbbf4be710cc8cbd95d20b24c4
    """
    out = ctypes.c_int(0)
    call_from_clib("get_native_id", ctypes.pointer(out), index, clib_prefix="afcu")
    return out.value


def get_stream(index: int) -> int:
    """
    source: https://arrayfire.org/docs/group__cuda__mat.htm#ga8323b850f80afe9878b099f647b0a7e5
    """
    out = AFArray.create_null_pointer()
    call_from_clib("get_stream", ctypes.pointer(out), index, clib_prefix="afcu")
    return out.value  # type: ignore[return-value]


def set_native_id(index: int) -> None:
    """
    source: https://arrayfire.org/docs/group__cuda__mat.htm#ga966f4c6880e90ce91d9599c90c0db378
    """
    call_from_clib("set_native_id", index, clib_prefix="afcu")
    return None
