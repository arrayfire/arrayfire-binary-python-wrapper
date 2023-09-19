import ctypes

from arrayfire_wrapper.backend import _backend
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._error_handler import safe_call


def flat(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__flat.htm#gac6dfb22cbd3b151ddffb9a4ddf74455e
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_flat(ctypes.pointer(out), arr))
    return out


def flip(arr: AFArray, dim: int = 0, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__flip.htm#gac0795e2a4343ea8f897b3b7d23802ccb
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_flip(ctypes.pointer(out), arr, ctypes.c_int(dim)))
    return out


def join(dim: int, first: AFArray, second: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__join.htm#ga4c0b185d13b49023cc22c0269eedbdb2
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_join(ctypes.pointer(out), dim, first, second))
    return out


def join_many(dim: int, n_arrays: int, *inputs: AFArray) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__join.htm#ga67a36384247f6bb40254e0cb2e6d5d5c
    """
    out = AFArray.create_null_pointer()

    size = 10 if len(inputs) > 10 else len(inputs)  # NOTE: API limitations
    inputs_arr = ctypes.c_void_p * size
    inputs_arr_vector = inputs_arr(*inputs[:size])

    safe_call(_backend.clib.af_join_many(ctypes.pointer(out), dim, n_arrays, ctypes.pointer(inputs_arr_vector)))
    return out
