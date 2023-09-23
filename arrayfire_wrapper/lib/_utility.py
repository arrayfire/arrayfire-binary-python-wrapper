from __future__ import annotations

import ctypes
from typing import Any

from arrayfire_wrapper._backend import Backend, get_backend
from arrayfire_wrapper.defines import AFArray, CDimT
from arrayfire_wrapper.dtypes import to_str
from arrayfire_wrapper.lib._broadcast import bcast_var
from arrayfire_wrapper.lib._constants import ErrorCodes


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
    c_err = getattr(backend.clib, f"{clib_prefix}_{func_name}")(*args, **kwargs)

    if c_err == ErrorCodes.NONE.value:
        return

    _process_error(backend)


def _process_error(backend: Backend) -> None:
    err_str = ctypes.c_char_p(0)
    err_len = CDimT(0)
    backend.clib.af_get_last_error(ctypes.pointer(err_str), ctypes.pointer(err_len))
    raise RuntimeError(to_str(err_str))
