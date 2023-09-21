import ctypes
from enum import Enum

from arrayfire_wrapper._backend import _backend
from arrayfire_wrapper.defines import CDimT
from arrayfire_wrapper.dtypes import to_str


class _ErrorCodes(Enum):
    none = 0


def safe_call(c_err: int) -> None:
    if c_err == _ErrorCodes.none.value:
        return

    err_str = ctypes.c_char_p(0)
    err_len = CDimT(0)
    _backend.clib.af_get_last_error(ctypes.pointer(err_str), ctypes.pointer(err_len))
    raise RuntimeError(to_str(err_str))
