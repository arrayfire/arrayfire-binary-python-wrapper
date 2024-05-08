import ctypes

import numpy as np
import pyopencl.array as cl  # type: ignore[import-untyped]

from arrayfire_wrapper.defines import AFArray, CShape
from arrayfire_wrapper.dtypes import c_api_value_to_dtype, str_to_dtype
from arrayfire_wrapper.lib._utility import call_from_clib
from arrayfire_wrapper.lib.create_and_modify_array.manage_array import create_array, get_data_ptr, get_dims, get_type


def numpy_to_af_array(np_arr: np.ndarray) -> AFArray:
    out = AFArray(0)
    shape = np_arr.shape
    c_shape = CShape(*shape)

    c_type = np.ctypeslib.as_ctypes_type(np_arr.dtype)
    dtype = str_to_dtype(c_type)

    call_from_clib(
        create_array.__name__,
        ctypes.pointer(out),
        np_arr.ctypes.data_as(ctypes.c_void_p),
        c_shape.original_shape,
        ctypes.pointer(c_shape.c_array),
        dtype.c_api_value,
    )
    return out


def af_to_numpy_array(af_arr: AFArray) -> np.ndarray:
    shape = get_dims(af_arr)
    dtype = c_api_value_to_dtype(get_type(af_arr))
    typecode = dtype.typecode

    out = np.empty(shape, typecode, "F")
    call_from_clib(get_data_ptr.__name__, ctypes.c_void_p(out.ctypes.data), af_arr)
    return out


def pyopencl_to_af_array(pycl_arr: cl.Array) -> AFArray:
    out = AFArray(0)
    np_arr = pycl_arr.get()

    shape = np_arr.shape
    c_shape = CShape(*shape)

    c_type = np.ctypeslib.as_ctypes_type(np_arr.dtype)
    dtype = str_to_dtype(c_type)

    call_from_clib(
        create_array.__name__,
        ctypes.pointer(out),
        np_arr.ctypes.data_as(ctypes.c_void_p),
        c_shape.original_shape,
        ctypes.pointer(c_shape.c_array),
        dtype.c_api_value,
    )

    return out
