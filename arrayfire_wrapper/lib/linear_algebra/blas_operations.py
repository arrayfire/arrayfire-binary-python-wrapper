import ctypes
from typing import Any

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.dtypes import c_api_value_to_dtype, complex32, complex64, float32, float64
from arrayfire_wrapper.lib._constants import MatProp
from arrayfire_wrapper.lib._utility import call_from_clib
from arrayfire_wrapper.lib.create_and_modify_array.manage_array import get_type, copy_array


def dot(lhs: AFArray, rhs: AFArray, lhs_opts: MatProp, rhs_opts: MatProp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__blas__func__dot.htm#ga030ea5d9b694a4d3847f69254ab4a90d
    """
    out = AFArray.create_null_pointer()
    call_from_clib(dot.__name__, ctypes.pointer(out), lhs, rhs, lhs_opts.value, rhs_opts.value)
    return out


def dot_all(lhs: AFArray, rhs: AFArray, lhs_opts: MatProp, rhs_opts: MatProp, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__blas__func__dot.htm#gafb619ba32e85dfac62237929da911995
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(
        dot_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), lhs, rhs, lhs_opts.value, rhs_opts.value
    )
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def gemm(lhs: AFArray, rhs: AFArray, lhs_opts: MatProp, rhs_opts: MatProp, alpha: Any, beta: Any, accum: AFArray | None, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__blas__func__matmul.htm#ga0463ae584163128718237b02faf5caf7
    """
    out = None
    if not accum is None:
        out = copy_array(accum)
    else:
        beta = 0.0
        out = AFArray.create_null_pointer()

    lhs_dtype = c_api_value_to_dtype(get_type(lhs))

    type_mapping = {
        float32: (ctypes.c_float, _af_cfloat),
        complex32: (_af_cfloat, _af_cfloat),
        float64: (ctypes.c_double, _af_cdouble),
        complex64: (_af_cdouble, _af_cdouble),
    }

    if lhs_dtype in type_mapping:
        alpha_ptr = _cast_to_void_ptr(alpha, type_mapping[lhs_dtype][0])
        beta_ptr = _cast_to_void_ptr(beta, type_mapping[lhs_dtype][1])
    else:
        raise TypeError(f"{lhs_dtype.name} is currently unsupported as input type for gemm().")

    call_from_clib(gemm.__name__, ctypes.pointer(out), lhs_opts.value, rhs_opts.value, alpha_ptr, lhs, rhs, beta_ptr)
    return out


def matmul(lhs: AFArray, rhs: AFArray, lhs_opts: MatProp, rhs_opts: MatProp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__blas__func__matmul.htm#ga3f3f29358b44286d19ff2037547649fe
    """
    out = AFArray.create_null_pointer()
    call_from_clib(matmul.__name__, ctypes.pointer(out), lhs, rhs, lhs_opts.value, rhs_opts.value)
    return out


class _af_cfloat(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class _af_cdouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


def _cast_to_void_ptr(value, data_type):  # type: ignore[no-untyped-def]
    """
    Casts a given value to a ctypes void pointer based on the specified data type.
    """
    if isinstance(value, data_type):
        return ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)
    elif isinstance(value, tuple):
        return ctypes.cast(ctypes.pointer(data_type(*value)), ctypes.c_void_p)
    else:
        return ctypes.cast(ctypes.pointer(data_type(value)), ctypes.c_void_p)
