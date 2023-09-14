from __future__ import annotations

import ctypes
from collections.abc import Callable

from arrayfire_wrapper._typing import AFArray
from arrayfire_wrapper.library._broadcast import bcast_var
from arrayfire_wrapper.library._error_handler import safe_call


def binary_op(c_func: Callable, lhs: AFArray, rhs: AFArray, /) -> AFArray:
    out = AFArray(0)
    safe_call(c_func(ctypes.pointer(out), lhs, rhs, bcast_var.get()))
    return out


def unary_op(c_func: Callable, arr: AFArray, /) -> AFArray:
    out = AFArray(0)
    safe_call(c_func(ctypes.pointer(out), arr))
    return out
