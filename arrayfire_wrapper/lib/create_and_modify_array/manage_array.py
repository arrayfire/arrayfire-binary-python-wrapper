import ctypes
from typing import cast

from arrayfire_wrapper._backend import _backend
from arrayfire_wrapper.defines import AFArray, ArrayBuffer, CDimT, CShape
from arrayfire_wrapper.dtypes import Dtype

from .._error_handler import safe_call


def copy_array(arr: AFArray, /) -> AFArray:
    """
    This function takes an `AFArray` object as input and returns a new `AFArray` object
    that is a copy of the input array on a C pointer.

    Parameters
    ----------
    arr : AFArray
        The input `AFArray` object.

    Returns
    -------
    AFArray
        A new `AFArray` object that is a copy of the input array.

    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6040dc6f0eb127402fbf62c1165f0b9d
    """
    out = AFArray(0)
    safe_call(_backend.clib.af_copy_array(ctypes.pointer(out), arr))
    return out


def create_array(shape: tuple[int, ...], dtype: Dtype, array_buffer: ArrayBuffer, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga834be32357616d8ab735087c6f681858
    """
    out = AFArray(0)
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_create_array(
            ctypes.pointer(out),
            ctypes.c_void_p(array_buffer.address),
            c_shape.original_shape,
            ctypes.pointer(c_shape.c_array),
            dtype.c_api_value,
        )
    )
    return out


def create_handle(shape: tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga3b8f5cf6fce69aa1574544bc2d44d7d0
    """
    out = AFArray(0)
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_create_handle(
            ctypes.pointer(out), c_shape.original_shape, ctypes.pointer(c_shape.c_array), dtype.c_api_value
        )
    )
    return out


def device_array(shape: tuple[int, ...], dtype: Dtype, array_buffer: ArrayBuffer, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaad4fc77f872217e7337cb53bfb623cf5
    """
    out = AFArray(0)
    c_shape = CShape(*shape)

    safe_call(
        _backend.clib.af_device_array(
            ctypes.pointer(out),
            ctypes.c_void_p(array_buffer.address),
            c_shape.original_shape,
            ctypes.pointer(c_shape.c_array),
            dtype.c_api_value,
        )
    )
    return out


def eval(arr: AFArray, /) -> None:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga9de141bfc5936741d2496f59c1bac777
    """
    safe_call(_backend.clib.af_eval(arr))
    return None


# TODO Discussion needed
def eval_multiple(arr: AFArray, data: int, /) -> None:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga9e08f4cda2471a477d2fa91c2356f72c
    """
    safe_call(_backend.clib.af_eval_multiple(ctypes.c_int(data), arr))
    return None


# HACK does not fit the original API. Discussion needed
def get_data_ptr(arr: AFArray, size: int, dtype: Dtype, /) -> ctypes.Array:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6040dc6f0eb127402fbf62c1165f0b9d
    """
    c_shape = dtype.c_type * size
    ctypes_array = c_shape()
    safe_call(_backend.clib.af_get_data_ptr(ctypes.pointer(ctypes_array), arr))
    return ctypes_array


def get_data_ref_count() -> int:
    # FIXME
    return NotImplemented


def get_dims(arr: AFArray, /) -> tuple[int, ...]:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga8b90da50a532837d9763e301b2267348
    """
    d0 = CDimT(0)
    d1 = CDimT(0)
    d2 = CDimT(0)
    d3 = CDimT(0)

    safe_call(
        _backend.clib.af_get_dims(ctypes.pointer(d0), ctypes.pointer(d1), ctypes.pointer(d2), ctypes.pointer(d3), arr)
    )
    return (d0.value, d1.value, d2.value, d3.value)


def get_elements(arr: AFArray, /) -> int:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6845bbe4385a60a606b88f8130252c1f
    """
    out = CDimT(0)

    safe_call(_backend.clib.af_get_elements(ctypes.pointer(out), arr))
    return out.value


def get_manual_eval_flag() -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6845bbe4385a60a606b88f8130252c1f
    """
    out = ctypes.c_bool(False)
    safe_call(_backend.clib.af_get_manual_eval_flag(ctypes.pointer(out)))
    return bool(out.value)


def get_numdims(arr: AFArray, /) -> int:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaefa019d932ff58c2a829ce87edddd2a8
    """
    out = ctypes.c_uint(0)

    safe_call(_backend.clib.af_get_numdims(ctypes.pointer(out), arr))
    return out.value


def get_scalar(arr: AFArray, dtype: Dtype, /) -> int | float | complex | bool | None:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaefe2e343a74a84bd43b588218ecc09a3
    """
    out = dtype.c_type()
    safe_call(_backend.clib.af_get_scalar(ctypes.pointer(out), arr))
    return cast(int | float | complex | bool | None, out.value)


def get_type(arr: AFArray, /) -> int:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga0dda6898e1c0d9a43efb56cd6a988c9b
    """
    out = ctypes.c_int(0)

    safe_call(_backend.clib.af_get_type(ctypes.pointer(out), arr))
    return out.value


def is_bool(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gafae10fc1378b72404120572e21ff5d27
    """
    out = ctypes.c_bool(False)
    safe_call(_backend.clib.af_is_bool(ctypes.pointer(out), arr))
    return bool(out.value)


def is_column(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga184b5a5feb146f2a2a44fed36b35e435
    """
    out = ctypes.c_bool(False)
    safe_call(_backend.clib.af_is_column(ctypes.pointer(out), arr))
    return bool(out.value)


def is_complex(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gacd8a5edf6467340d0b40341be9f677e9
    """
    out = ctypes.c_bool(False)
    safe_call(_backend.clib.af_is_complex(ctypes.pointer(out), arr))
    return bool(out.value)


def is_double(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gadb499641bfc02dfa56a75be9ba237e3f
    """
    out = ctypes.c_bool(False)
    safe_call(_backend.clib.af_is_double(ctypes.pointer(out), arr))
    return bool(out.value)


def is_empty(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga19c749e95314e1c77d816ad9952fb680
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_empty(ctypes.pointer(out), arr))
    return bool(out.value)


def is_floating(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga5eb0d277412a2beeffe7e7a9f89b98ea
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_floating(ctypes.pointer(out), arr))
    return bool(out.value)


def is_half(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga247a830d52f1cb2450369af3d8d8f2f1
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_half(ctypes.pointer(out), arr))
    return bool(out.value)


def is_integer(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga768e8326a6aaa81c6543949afc781af2
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_integer(ctypes.pointer(out), arr))
    return bool(out.value)


def is_real(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gafaa0c1597ef34a7320ed589f80fdce10
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_real(ctypes.pointer(out), arr))
    return bool(out.value)


def is_realfloating(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6f2b3e689d07f5135dfc1ee6cf9825a3
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_realfloating(ctypes.pointer(out), arr))
    return bool(out.value)


def is_row(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gabbe3249a355293daabd5907d7df67c6a
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_row(ctypes.pointer(out), arr))
    return bool(out.value)


def is_scalar(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gae3dfa6dc1c03c5efe7482bfc9c41266a
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_scalar(ctypes.pointer(out), arr))
    return bool(out.value)


def is_single(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga1bd444b2c78a4f4731d4523a90200175
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_single(ctypes.pointer(out), arr))
    return bool(out.value)


def is_sparse(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gac96d3ca37a435874de22a76838a8cf54
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_sparse(ctypes.pointer(out), arr))
    return bool(out.value)


def is_vector(arr: AFArray, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga00a23c7dd281fdcdec10b8395e611154
    """
    out = ctypes.c_bool()
    safe_call(_backend.clib.af_is_vector(ctypes.pointer(out), arr))
    return bool(out.value)


def release_array(arr: AFArray, /) -> None:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gad6c58648ed0db398e170dabf045e8309
    """
    safe_call(_backend.clib.af_release_array(arr))


def retain_array(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga7ed45b3f881c0f6c80c5cf2af886dbab
    """
    out = AFArray(0)
    safe_call(_backend.clib.af_retain_array(ctypes.pointer(out), arr))
    return out


def set_manual_eval_flag(flag: bool, /) -> None:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga869f5e9331e9e010345de2589add7bae
    """
    safe_call(_backend.clib.af_set_manual_eval_flag(flag))
    return None


def write_array() -> None:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gafef13633d184acc726ad9daca7a7bc99
    """
    # FIXME
    return None
