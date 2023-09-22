from __future__ import annotations

import ctypes

from arrayfire_wrapper.defines import AFArray, CShape
from arrayfire_wrapper.dtypes import Dtype
from arrayfire_wrapper.lib._utility import call_from_clib


class AFRandomEngineHandle(ctypes.c_void_p):
    @classmethod
    def create_null_pointer(cls) -> AFRandomEngineHandle:
        cls.value = 0
        return cls()


def create_random_engine(engine_type: int, seed: int, /) -> AFRandomEngineHandle:
    out = AFRandomEngineHandle.create_null_pointer()
    call_from_clib(create_random_engine.__name__, ctypes.pointer(out), engine_type, ctypes.c_longlong(seed))
    return out


def release_random_engine(engine: AFRandomEngineHandle, /) -> None:
    call_from_clib(release_random_engine.__name__, engine)
    return None


def random_engine_set_type(engine: AFRandomEngineHandle, engine_type: int, /) -> None:
    call_from_clib(random_engine_set_type.__name__, ctypes.pointer(engine), engine_type)
    return None


def random_engine_get_type(engine: AFRandomEngineHandle, /) -> int:
    out = ctypes.c_int(0)
    call_from_clib(random_engine_get_type.__name__, ctypes.pointer(out), engine)
    return out.value


def random_engine_set_seed(engine: AFRandomEngineHandle, seed: int, /) -> None:
    call_from_clib(random_engine_set_seed.__name__, ctypes.pointer(engine), ctypes.c_longlong(seed))
    return None


def random_engine_get_seed(engine: AFRandomEngineHandle, /) -> int:
    out = ctypes.c_longlong(0)
    call_from_clib(random_engine_get_seed.__name__, ctypes.pointer(out), engine)
    return out.value


def randu(shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__random__func__randu.htm#ga412e2c2f5135bdda218c3487c487d3b5
    """
    out = AFArray.create_null_pointer()
    c_shape = CShape(*shape)
    call_from_clib(randu.__name__, ctypes.pointer(out), 4, c_shape.c_array, dtype.c_api_value)
    return out


def random_uniform(shape: tuple[int, ...], dtype: Dtype, engine: AFRandomEngineHandle, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__random__func__randu.htm#ga2ca76d970cfac076f9006755582a4a4c
    """
    out = AFArray.create_null_pointer()
    c_shape = CShape(*shape)
    call_from_clib(random_uniform.__name__, ctypes.pointer(out), 4, c_shape.c_array, dtype.c_api_value, engine)
    return out
