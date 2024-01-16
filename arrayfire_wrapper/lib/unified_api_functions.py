import ctypes

from arrayfire_wrapper._backend import BackendType
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def get_active_backend() -> str:
    """
    source: https://arrayfire.org/docs/group__unified__func__getactivebackend.htm#gac6a8e976a151d007e0cf5cf4f11da2a9
    """
    out = ctypes.c_int(0)
    call_from_clib(get_active_backend.__name__, ctypes.pointer(out))
    return BackendType(out.value).name


def get_available_backends() -> list[int]:
    """
    source: https://arrayfire.org/docs/group__unified__func__getavailbackends.htm#ga92a9ce85385763bfa83911cda905afe8
    """
    out = ctypes.c_int(0)
    call_from_clib(get_available_backends.__name__, ctypes.pointer(out))
    rv = out.value
    return [bt.value & rv for bt in BackendType]


def get_backend_count() -> int:
    """
    source: https://arrayfire.org/docs/group__unified__func__getbackendcount.htm#gad38c2dfedfdabfa264afa46d8664e9cd
    """
    out = ctypes.c_uint(0)
    call_from_clib(get_backend_count.__name__, ctypes.pointer(out))
    return out.value


def get_backend_id(arr: AFArray, /) -> int:
    """
    source: https://arrayfire.org/docs/group__unified__func__getbackendid.htm#ga5fc39e209e1886cf250aec265c0d9079
    """
    out = ctypes.c_int(0)
    call_from_clib(get_backend_id.__name__, ctypes.pointer(out), arr)
    return out.value


def get_device_id(arr: AFArray, /) -> int:
    """
    source: https://arrayfire.org/docs/group__unified__func__getdeviceid.htm#ga5d94b64dccd1c7cbc7a3a69fa64888c3
    """
    out = ctypes.c_int(0)
    call_from_clib(get_device_id.__name__, ctypes.pointer(out), arr)
    return out.value


def set_backend(backend: BackendType, /) -> None:
    """
    source: https://arrayfire.org/docs/group__unified__func__setbackend.htm#ga6fde820e8802776b7fc823504b37f1b4
    """
    call_from_clib(set_backend.__name__, backend.value)
