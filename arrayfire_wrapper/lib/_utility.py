from __future__ import annotations

import ctypes
from typing import Any

from arrayfire_wrapper._backend import get_backend
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._broadcast import bcast_var
from arrayfire_wrapper.lib._error_handler import safe_call


def binary_op(c_func_name: str, lhs: AFArray, rhs: AFArray, /) -> AFArray:
    out = AFArray.create_null_pointer()
    call_from_clib(c_func_name, ctypes.pointer(out), lhs, rhs, bcast_var.get())
    return out


def unary_op(c_func_name: str, arr: AFArray, /) -> AFArray:
    out = AFArray.create_null_pointer()
    call_from_clib(c_func_name, ctypes.pointer(out), arr)
    return out


def call_from_clib(func_name: str, *args: Any, clib_prefix: str = "af", **kwargs: Any) -> None:
    backend = get_backend()
    safe_call(getattr(backend.clib, f"{clib_prefix}_{func_name}")(*args, **kwargs))
