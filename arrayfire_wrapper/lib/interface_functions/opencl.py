import ctypes
from enum import Enum

from arrayfire_wrapper.lib._utility import call_from_clib


class DeviceType(Enum):
    CPU = 2
    GPU = 4
    ACC = 8
    UNKNOWN = -1


class PlatformType(Enum):
    AMD = 0
    APPLE = 1
    INTEL = 2
    NVIDIA = 3
    BEIGNET = 4
    POCL = 5
    UNKNOWN = -1


def get_context(retain: bool = False) -> ctypes.c_void_p:
    """
    source: https://arrayfire.org/docs/group__opencl__mat.htm#gad42de383f405b3e38d6eb669c0cbe2e3
    """
    out = ctypes.c_void_p()
    call_from_clib(get_context.__name__, ctypes.pointer(out), retain, clib_prefix="afcl")
    return out  # type: ignore[return-value]


def get_queue(retain: bool = False) -> ctypes.c_void_p:
    """
    source: https://arrayfire.org/docs/group__opencl__mat.htm#gab1701ef4f2b68429eb31c1e21c88d0bc
    """
    out = ctypes.c_void_p()
    call_from_clib(get_queue.__name__, ctypes.pointer(out), retain, clib_prefix="afcl")
    return out  # type: ignore[return-value]


def get_device_id() -> int:
    """
    source: https://arrayfire.org/docs/group__opencl__mat.htm#gaf7258055284e65a8647a49c3f3b9feee
    """
    out = ctypes.c_void_p()
    call_from_clib(get_device_id.__name__, ctypes.pointer(out), clib_prefix="afcl")
    return out  # type: ignore[return-value]


def set_device_id(idx: int) -> None:
    """
    source: https://arrayfire.org/docs/group__opencl__mat.htm#ga600361a20ceac2a65590b67fc0366314
    """
    call_from_clib(set_device_id.__name__, ctypes.c_int64(idx), clib_prefix="afcl")
    return None


def add_device_context(dev: int, ctx: int, que: int) -> None:
    """
    source: https://arrayfire.org/docs/group__opencl__mat.htm#ga49f596a4041fb757f1f5a75999cf8858
    """
    call_from_clib(
        add_device_context.__name__, ctypes.c_int64(dev), ctypes.c_int64(ctx), ctypes.c_int64(que), clib_prefix="afcl"
    )
    return None


def set_device_context(dev: int, ctx: int) -> None:
    """
    source: https://arrayfire.org/docs/group__opencl__mat.htm#ga975661f2b06dddb125c5d1757160b02c
    """
    call_from_clib(set_device_context.__name__, ctypes.c_int64(dev), ctypes.c_int64(ctx), clib_prefix="afcl")
    return None


def delete_device_context(dev: int, ctx: int) -> None:
    """
    source: https://arrayfire.org/docs/group__opencl__mat.htm#ga1a56dcf05099d6ac0a3b7701f7cb23f8
    """
    call_from_clib(delete_device_context.__name__, ctypes.c_int64(dev), ctypes.c_int64(ctx), clib_prefix="afcl")
    return None


def get_device_type() -> DeviceType:
    """
    source: https://arrayfire.org/docs/group__opencl__mat.htm#ga5e360e0fe0eb55d0046191bc3fd6f81d
    """
    res = ctypes.c_void_p()
    call_from_clib(get_device_type.__name__, ctypes.pointer(res), clib_prefix="afcl")
    return DeviceType(res.value)


def get_platform() -> PlatformType:
    """
    source: https://arrayfire.org/docs/group__opencl__mat.htm#ga5e360e0fe0eb55d0046191bc3fd6f81d&gsc.tab=0
    """
    res = ctypes.c_void_p()
    call_from_clib(get_platform.__name__, ctypes.pointer(res), clib_prefix="afcl")
    return PlatformType(res.value)
