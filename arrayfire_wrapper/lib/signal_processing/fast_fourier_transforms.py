import ctypes

from arrayfire_wrapper.defines import AFArray, CDimT
from arrayfire_wrapper.lib._utility import call_from_clib


def fft(arr: AFArray, norm_factor: float, dim0: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft.htm#ga64d0db9e59c9410ba738591ad146a884
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor), CDimT(dim0))
    return out


def fft_inplace(arr: AFArray, norm_factor: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft.htm#gaa2f03c9ee1cb80dc184c0b0a13176da1
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft_inplace.__name__, arr, ctypes.c_double(norm_factor))
    return out


def set_fft_plan_cache_size(cache_size: int, /) -> None:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft.htm#ga4ddef19b43d9a50c97b1a835df60279a
    """
    call_from_clib(set_fft_plan_cache_size.__name__, ctypes.c_size_t(cache_size))
    return None


def fft2(arr: AFArray, norm_factor: float, dim0: int, dim1: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft2.htm#gaab3fb1ed398e208a615036b4496da611
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft2.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor), CDimT(dim0), CDimT(dim1))
    return out


def fft2_inplace(arr: AFArray, norm_factor: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft2.htm#gacdeebb3f221ae698833dc4900a172b8c
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft2_inplace.__name__, arr, ctypes.c_double(norm_factor))
    return out


def fft3(arr: AFArray, norm_factor: float, dim0: int, dim1: int, dim2: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft3.htm#ga5138ef1740ece0fde2c796904d733c12
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        fft3.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor), CDimT(dim0), CDimT(dim1), CDimT(dim2)
    )
    return out


def fft3_inplace(arr: AFArray, norm_factor: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft3.htm#ga0b0ab1facc734503bebc1670c4826646
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft3_inplace.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor))
    return out


def convert_to_c2r_dim(dim: int, is_odd: bool, /) -> int:
    return 2 * (dim - 1) + int(is_odd)


def fft_c2r(arr: AFArray, norm_factor: float, is_odd: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft__c2r.htm#gaa5efdfd84213a4a07d81a5d534cde5ac
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft_c2r.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor), is_odd)
    return out


def fft2_c2r(arr: AFArray, norm_factor: float, is_odd: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft__c2r.htm#gaaa7da16f226cacaffced631e08da4493
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft2_c2r.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor), is_odd)
    return out


def fft3_c2r(arr: AFArray, norm_factor: float, is_odd: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft__c2r.htm#gaa9b3322d9ffab15268919e1f114bed24
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft3_c2r.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor), is_odd)
    return out


def fft_r2c(arr: AFArray, norm_factor: float, pad0: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft__r2c.htm#ga7486f342182a18e773f14cc2ab4cb551
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft_r2c.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor), CDimT(pad0))
    return out


def fft2_r2c(arr: AFArray, norm_factor: float, pad0: int, pad1: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft__r2c.htm#ga90b2f78dc0ed69867145c71104bd063f
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft2_r2c.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor), CDimT(pad0), CDimT(pad1))
    return out


def fft3_r2c(arr: AFArray, norm_factor: float, pad0: int, pad1: int, pad2: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__fft__r2c.htm#gab4ca074b54218b74d8cfbda63d38be51
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        fft3_r2c.__name__,
        ctypes.pointer(out),
        arr,
        ctypes.c_double(norm_factor),
        CDimT(pad0),
        CDimT(pad1),
        CDimT(pad2),
    )
    return out


def ifft(arr: AFArray, norm_factor: float, dim0: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__ifft.htm#ga2d62c120b474b3b937b0425c994645fe
    """
    out = AFArray.create_null_pointer()
    call_from_clib(ifft.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor), CDimT(dim0))
    return out


def ifft_inplace(arr: AFArray, norm_factor: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__ifft.htm#ga827379bef0e2cadb382c1b6301c91429
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft_inplace.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor))
    return out


def ifft2(arr: AFArray, norm_factor: float, dim0: int, dim1: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__ifft2.htm#ga7cd29c6a35c19240635b62cc5c30dc4f
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft2.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor), CDimT(dim0), CDimT(dim1))
    return out


def ifft2_inplace(arr: AFArray, norm_factor: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__ifft2.htm#ga9e6a165d44306db4552a56d421ce5d05
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft2_inplace.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor))
    return out


def ifft3(arr: AFArray, norm_factor: float, dim0: int, dim1: int, dim2: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__ifft3.htm#gafdabcf20f793430134550e37f7a71bbd
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        fft3.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor), CDimT(dim0), CDimT(dim1), CDimT(dim2)
    )
    return out


def ifft3_inplace(arr: AFArray, norm_factor: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__ifft3.htm#ga439a7a49723bc6cf77cf4fe7f8dfe334
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft3_inplace.__name__, ctypes.pointer(out), arr, ctypes.c_double(norm_factor))
    return out
