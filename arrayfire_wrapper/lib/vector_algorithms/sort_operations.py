import ctypes

from arrayfire_wrapper._backend import _backend
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._error_handler import safe_call


def sort(arr: AFArray, dim: int, is_ascending: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__sort__func__sort.htm#gac4460d605452515d07ee8432f906aa8e
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_sort(ctypes.pointer(out), arr, ctypes.c_uint(dim), ctypes.c_bool(is_ascending)))
    return out


def sort_by_key(keys: AFArray, values: AFArray, dim: int, is_ascending: bool, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__sort__func__sort__keys.htm#ga7d4fcaf229ece5fbbe30a638d9a60b8a
    """
    out_keys = AFArray.create_null_pointer()
    out_values = AFArray.create_null_pointer()
    safe_call(
        _backend.clib.af_sort_by_key(
            ctypes.pointer(out_keys),
            ctypes.pointer(out_values),
            keys,
            values,
            ctypes.c_uint(dim),
            ctypes.c_bool(is_ascending),
        )
    )
    return (out_keys, out_values)


def sort_index(arr: AFArray, dim: int, is_ascending: bool, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__sort__func__sort__index.htm#ga55675cd825c320db87398b1010b6ae41
    """
    out = AFArray.create_null_pointer()
    out_idx = AFArray.create_null_pointer()
    safe_call(
        _backend.clib.af_sort_index(
            ctypes.pointer(out), ctypes.pointer(out_idx), arr, ctypes.c_uint(dim), ctypes.c_bool(is_ascending)
        )
    )
    return (out, out_idx)
