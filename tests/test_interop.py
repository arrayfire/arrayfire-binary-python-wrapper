import numpy as np
import pyopencl as cl  # type: ignore
import pyopencl.array as cl_array  # type: ignore

import arrayfire_wrapper.lib as wrapper
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.dtypes import int16
from arrayfire_wrapper.lib.create_and_modify_array.manage_array import get_dims, get_numdims
from arrayfire_wrapper.lib.interface_functions.interop import (  # noqa: E501
    af_to_numpy_array,
    numpy_to_af_array,
    pyopencl_to_af_array,
)

# flake8: noqa: E203


def test_numpy_to_af_array_type() -> None:
    arr = np.array([1, 2, 3, 4])

    af_array = numpy_to_af_array(arr)

    assert isinstance(af_array, AFArray)


def test_af_to_numpy_array_type() -> None:
    arr = wrapper.constant(2, (5, 5), int16)

    np_arr = af_to_numpy_array(arr)

    assert isinstance(np_arr, np.ndarray)


def test_pyopencl_to_af_array_type() -> None:
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    host_array = np.array([1, 2, 3, 4])

    cl_array_device = cl_array.to_device(queue, host_array)

    af_array = pyopencl_to_af_array(cl_array_device)

    assert isinstance(af_array, AFArray)


def test_numpy_to_af_array_shape() -> None:
    np_arr = np.array([1, 2, 3, 4])

    af_arr = numpy_to_af_array(np_arr)

    assert get_dims(af_arr)[0 : get_numdims(af_arr)] == np_arr.shape[0 : get_numdims(af_arr)]


def test_af_to_numpy_array_shape() -> None:
    af_arr = wrapper.constant(2, (5, 5), int16)

    np_arr = af_to_numpy_array(af_arr)
    assert np_arr.shape[0 : get_numdims(af_arr)] == get_dims(af_arr)[0 : get_numdims(af_arr)]


def test_pyopencl_to_af_array_shape() -> None:
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    host_array = np.array([1, 2, 3, 4])

    cl_arr = cl_array.to_device(queue, host_array)

    af_arr = pyopencl_to_af_array(cl_arr)

    assert cl_arr.shape[0 : get_numdims(af_arr)] == get_dims(af_arr)[0 : get_numdims(af_arr)]
