import ctypes

from arrayfire_wrapper.defines import AFArray, CDimT
from arrayfire_wrapper.dtypes import to_str
from arrayfire_wrapper.lib._utility import call_from_clib


def alloc_host(num_bytes: int, /) -> int:
    """
    source: https://arrayfire.org/docs/group__device__func__alloc__host.htm#ga3218dbf32dc51436d9557cc73cda6579

    Allocate a buffer on the host with specified number of bytes.
    """
    out = AFArray.create_null_pointer()
    call_from_clib(alloc_host.__name__, ctypes.pointer(out), CDimT(num_bytes))
    return out.value  # type: ignore[return-value]


def alloc_device(num_bytes: int, /) -> int:
    # NOTE af_alloc_device is marked as deprecated, so used af_alloc_device_v2 instead
    """
    source: https://arrayfire.org/docs/group__device__func__alloc.htm#gaa8868199b29eae4bac42cc22ff5891a9

    Allocate a buffer on the device with specified number of bytes.
    """
    out = AFArray.create_null_pointer()
    call_from_clib("alloc_device_v2", ctypes.pointer(out), CDimT(num_bytes))
    return out.value  # type: ignore[return-value]


def device_info() -> dict[str, str]:
    """
    Returns a map with the following fields:
        - "device": Name of the current device.
        - "backend": The current backend being used.
        - "toolkit": The toolkit version for the backend.
        - "compute": The compute version of the device.

    source: https://arrayfire.org/docs/group__device__func__prop.htm#ga9ad045fab0fc6e4260a4d13881a1a5d9
    """
    out = {}
    c_char_256 = ctypes.c_char * 256

    device_name = c_char_256()
    backend_name = c_char_256()
    toolkit = c_char_256()
    compute = c_char_256()

    call_from_clib(
        device_info.__name__,
        ctypes.pointer(device_name),
        ctypes.pointer(backend_name),
        ctypes.pointer(toolkit),
        ctypes.pointer(compute),
    )

    out["device"] = to_str(device_name)
    out["backend"] = to_str(backend_name)
    out["toolkit"] = to_str(toolkit)
    out["compute"] = to_str(compute)

    return out


def device_gc() -> None:
    """
    source: https://arrayfire.org/docs/group__device__func__mem.htm#ga182a33d34b3288c5cf5b88cd02468c56

    Ask the garbage collector to free all unlocked memory
    """
    call_from_clib(device_gc.__name__)


def device_mem_info() -> dict[str, dict[str, int]]:
    """
    Returns a map with the following fields:
        - "alloc": Contains the map of the following
            - "buffers" : Total number of buffers allocated by memory manager.
            - "bytes"   : Total number of bytes allocated by memory manager.
        - "lock": Contains the map of the following
            - "buffers" : Total number of buffers currently in scope.
            - "bytes"   : Total number of bytes currently in scope.

    Note
    -----
    ArrayFire does not free memory when array goes out of scope. The memory is marked for reuse.
    - The difference between alloc buffers and lock buffers equals the number of free buffers.
    - The difference between alloc bytes and lock bytes equals the number of free bytes.

    source: https://arrayfire.org/docs/group__device__func__mem.htm#gae633760aed4638f8a5ea333e0774ac84

    """
    out = {}

    alloc_bytes = ctypes.c_size_t(0)
    alloc_buffers = ctypes.c_size_t(0)
    lock_bytes = ctypes.c_size_t(0)
    lock_buffers = ctypes.c_size_t(0)
    call_from_clib(
        device_mem_info.__name__,
        ctypes.pointer(alloc_bytes),
        ctypes.pointer(alloc_buffers),
        ctypes.pointer(lock_bytes),
        ctypes.pointer(lock_buffers),
    )

    out["alloc"] = {"buffers": alloc_buffers.value, "bytes": alloc_bytes.value}
    out["lock"] = {"buffers": lock_buffers.value, "bytes": lock_bytes.value}
    return out


def get_device_ptr(arr: AFArray) -> int:
    """
    Get the raw device pointer of an array

    Parameters
    ----------
    a: af.Array
       - A multi dimensional arrayfire array.

    Returns
    -------
        - internal device pointer held by a

    Note
    -----
        - The device pointer of `a` is not freed by memory manager until `unlock_device_ptr()` is called.
        - This function enables the user to interoperate arrayfire with other CUDA/OpenCL/C libraries.

    source: https://arrayfire.org/docs/group__device__func__mem.htm#ga58fda2d491cd27f31108e699b5aef506
    """
    out = AFArray.create_null_pointer()
    call_from_clib(get_device_ptr.__name__, ctypes.pointer(out), arr)
    return out.value  # type: ignore[return-value]


def get_kernel_cache_directory(length: int, path: str, /) -> None:
    """
    source: https://arrayfire.org/docs/group__device__func__mem.htm#ga0cca43230149189dcc46cceb5dba5588
    """
    call_from_clib(get_kernel_cache_directory.__name__, ctypes.c_size_t(length), path.encode("utf-8"))


def get_mem_step_size(step_bytes: int, /) -> None:
    """
    source: https://arrayfire.org/docs/group__device__func__mem.htm#ga4c04df1ae248a6a8aa0a28263323872a
    """
    call_from_clib(get_mem_step_size.__name__, ctypes.c_size_t(step_bytes))


def is_locked_array(arr: AFArray, /) -> bool:
    """
    Check if the input array is locked by the user.

    Parameters
    ----------
    a: af.Array
       - A multi dimensional arrayfire array.

    Returns
    -----------
    A bool specifying if the input array is locked.

    source: https://arrayfire.org/docs/group__device__func__mem.htm#gab99cb6a3744802742c98714fc88fb991
    """
    out = ctypes.c_bool(False)
    call_from_clib(is_locked_array.__name__, ctypes.pointer(out), arr)
    return bool(out.value)


def lock_array(arr: AFArray, /) -> None:
    """
    Ask arrayfire to not perform garbage collection on raw data held by an array.

    Parameters
    ----------
    a: af.Array
       - A multi dimensional arrayfire array.

    Note
    -----
        - The device pointer of `a` is not freed by memory manager until `unlock_array()` is called.

    source: https://arrayfire.org/docs/group__device__func__mem.htm#ga825e21412e9c8e3609c759f8106fd384
    """
    call_from_clib(lock_array.__name__, arr)


def lock_device_ptr(arr: AFArray, /) -> None:
    """
    source: https://arrayfire.org/docs/group__device__func__mem.htm#gac2ad5089cbca1a6cca8d87d42279c6a8
    """
    call_from_clib(lock_device_ptr.__name__, arr)


def print_mem_info(msg: str = "Memory Info", device_id: None | int = None, /) -> None:
    """
    Prints the memory used for the specified device.

    Parameters
    ----------
    title: optional. Default: "Memory Info"
       - Title to display before printing the memory info.
    device_id: optional. Default: None
       - Specifies the device for which the memory info should be displayed.
       - If None, uses the current device.

    Examples
    --------

    >>> a = af.randu(5,5)
    >>> af.print_mem_info()
    Memory Info
    ---------------------------------------------------------
    |     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |
    ---------------------------------------------------------
    |     0x706400000  |       1 KB |       Yes |        No |
    ---------------------------------------------------------
    >>> b = af.randu(5,5)
    >>> af.print_mem_info()
    Memory Info
    ---------------------------------------------------------
    |     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |
    ---------------------------------------------------------
    |     0x706400400  |       1 KB |       Yes |        No |
    |     0x706400000  |       1 KB |       Yes |        No |
    ---------------------------------------------------------
    >>> a = af.randu(1000,1000)
    >>> af.print_mem_info()
    Memory Info
    ---------------------------------------------------------
    |     POINTER      |    SIZE    |  AF LOCK  | USER LOCK |
    ---------------------------------------------------------
    |     0x706500000  |   3.815 MB |       Yes |        No |
    |     0x706400400  |       1 KB |       Yes |        No |
    |     0x706400000  |       1 KB |        No |        No |
    ---------------------------------------------------------

    source: https://arrayfire.org/docs/group__device__func__mem.htm#ga7c928031579de47fe21594fd745e9188
    """
    device_id = device_id if device_id else get_device()
    call_from_clib(print_mem_info.__name__, msg.encode("utf-8"), device_id)


def set_kernel_cache_directory(path: str, override_env: int, /) -> None:
    """
    source: https://arrayfire.org/docs/group__device__func__mem.htm#ga880be5cb0035d4f173d074ad06bce6a7
    """
    call_from_clib(set_kernel_cache_directory.__name__, path.encode("utf-8"), override_env)


def set_mem_step_size(step_bytes: int, /) -> None:
    """
    source: https://arrayfire.org/docs/group__device__func__mem.htm#ga3be9c5ea9ee828868f5d906333a11499
    """
    call_from_clib(set_mem_step_size.__name__, ctypes.c_size_t(step_bytes))


def unlock_array(arr: AFArray, /) -> None:
    """
    Tell arrayfire to resume garbage collection on raw data held by an array.

    Parameters
    ----------
    a: af.Array
       - A multi dimensional arrayfire array.

    source: https://arrayfire.org/docs/group__device__func__mem.htm#ga07151f8b3d69c1afe3cbd860fd98c36f
    """
    call_from_clib(unlock_array.__name__, arr)


def unlock_device_ptr(arr: AFArray, /) -> None:
    """
    source: https://arrayfire.org/docs/group__device__func__mem.htm#ga39817b0ba24db34f00c20cc3a20df6d4
    """
    call_from_clib(unlock_device_ptr.__name__, arr)
    return None


def free_host(pointer: int) -> None:
    """
    Free the host memory allocated by alloc_host

    source: https://arrayfire.org/docs/group__device__func__free__host.htm#ga3f1149a837a7ebbe8002d5d2244e3370
    """
    out = ctypes.c_void_p(pointer)
    call_from_clib(free_host.__name__, out)


def free_pinned(pointer: int) -> None:
    """
    Free the pinned memory allocated by alloc_pinned

    source: https://arrayfire.org/docs/group__device__func__free__pinned.htm#ga92ed71f45aa719b9be792afbab7415f2
    """
    out = ctypes.c_void_p(pointer)
    call_from_clib(free_pinned.__name__, out)


def free_device(pointer: int) -> None:
    # NOTE af_free_device is marked as deprecated, so used af_free_device_v2 instead
    """
    Free the device memory allocated by allooutice

    source: https://arrayfire.org/docs/group__device__func__free.htm#gadc0a469d9f5d885e73ee645a6dbf19f5
    """
    out = ctypes.c_void_p(pointer)
    call_from_clib(free_pinned.__name__, out)


def get_device_count() -> int:
    """
    Returns the number of devices available.

    source: https://arrayfire.org/docs/group__device__func__count.htm#ga0f163c809fb48e4cba530c6505f6e7b6
    """
    out = ctypes.c_int(0)
    call_from_clib(get_device_count.__name__, ctypes.pointer(out))
    return out.value


def info() -> None:
    """
    source: https://arrayfire.org/docs/group__device__func__info.htm#ga55e3054334c0fbc23676bc93a2bec066
    """
    call_from_clib(
        info.__name__,
    )
    return None


def init() -> None:
    """
    source: https://arrayfire.org/docs/group__device__func__info.htm#gacbdf7b79d778344d30deb77c06ac7367
    """
    call_from_clib(
        init.__name__,
    )
    return None


def info_string(verbose: bool = False, /) -> str:
    """
    source: https://arrayfire.org/docs/group__device__func__info__string.htm#gacc0f17e14982d390284347d4dc82b461
    """
    out = ctypes.c_char_p(0)
    call_from_clib(info_string.__name__, ctypes.pointer(out), verbose)
    return to_str(out)


def get_dbl_support(device_id: None | int = None, /) -> bool:
    """
    Check if double precision is supported on specified device.

    Parameters
    -----------
    device: optional: int. default: None.
         id of the desired device.

    Returns
    --------
        - True if double precision supported.
        - False if double precision not supported.

    source: https://arrayfire.org/docs/group__device__func__dbl.htm#ga71b5811b21be7a6d5e7fc0087ddf91c1
    """
    device_id = device_id if device_id else get_device()
    out = ctypes.c_bool(False)
    call_from_clib(get_dbl_support.__name__, ctypes.pointer(out), device_id)
    return bool(out.value)


def get_half_support(device_id: None | int = None, /) -> bool:
    """
    Check if half precision is supported on specified device.

    Parameters
    -----------
    device: optional: int. default: None.
         id of the desired device.

    Returns
    --------
        - True if half precision supported.
        - False if half precision not supported.

    source: https://arrayfire.org/docs/group__device__func__half.htm#ga83c2191dc82b2aba1d5f025abb769c3f
    """
    device_id = device_id if device_id else get_device()  # FIXME
    out = ctypes.c_bool(False)
    call_from_clib(get_half_support.__name__, ctypes.pointer(out), device_id)
    return bool(out.value)


def alloc_pinned(num_bytes: int, /) -> int:
    """
    Allocate a buffer on the host using pinned memory with specified number of bytes.

    source: https://arrayfire.org/docs/group__device__func__pinned.htm#ga0f8fd76dc179e7bd877e268a5579b215
    """
    out = ctypes.c_void_p(0)
    call_from_clib(alloc_pinned.__name__, ctypes.pointer(out), CDimT(num_bytes))
    return out.value  # type: ignore[return-value]


def get_device() -> int:
    """
    Returns the id of the current device.

    source: https://arrayfire.org/docs/group__device__func__set.htm#ga54120b126cfcb1b0b3ee25e0fc66b8a4
    """
    out = ctypes.c_int(0)
    call_from_clib(get_device.__name__, ctypes.pointer(out))
    return out.value


def set_device(device_id: int, /) -> None:
    """
    Change the active device to the specified id.

    Parameters
    -----------
    num: int.
         id of the desired device.
    """
    call_from_clib(set_device.__name__, device_id)
