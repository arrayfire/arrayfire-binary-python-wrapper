import ctypes

from arrayfire_wrapper.defines import AFArray, CShape
from arrayfire_wrapper.lib._constants import ConvDomain, ConvMode
from arrayfire_wrapper.lib._utility import call_from_clib


def convolve1(signal: AFArray, kernel: AFArray, mode: ConvMode, domain: ConvDomain, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__convolve1.htm#ga25d77b794463b5cd72cd0b7f4af140d7
    """
    out = AFArray.create_null_pointer()
    call_from_clib(convolve1.__name__, ctypes.pointer(out), signal, kernel, mode.value, domain.value)
    return out


def fft_convolve1(signal: AFArray, kernel: AFArray, mode: ConvMode, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__convolve1.htm#ga3cbc675cc70478f73803e906253a52c1
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft_convolve1.__name__, ctypes.pointer(out), signal, kernel, mode.value)
    return out


def convolve2(signal: AFArray, kernel: AFArray, mode: ConvMode, domain: ConvDomain, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__convolve2.htm#gaa6ab9a3d438ff793f530193b8ccb8003
    """
    out = AFArray.create_null_pointer()
    call_from_clib(convolve2.__name__, ctypes.pointer(out), signal, kernel, mode.value, domain.value)
    return out


def fft_convolve2(signal: AFArray, kernel: AFArray, mode: ConvMode, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__convolve2.htm#gab52ebe631d8358cdef1b5c8a95550556
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft_convolve2.__name__, ctypes.pointer(out), signal, kernel, mode.value)
    return out


def convolve2_nn(
    signal: AFArray, kernel: AFArray, stride: tuple[int, int], padding: tuple[int, int], dilation: tuple[int, int], /
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__convolve2.htm#ga06948be57cd0ec2e3646a7a57a5309b6
    """
    out = AFArray.create_null_pointer()
    stride_c_shape = CShape(*stride)
    padding_c_shape = CShape(*padding)
    dilation_c_shape = CShape(*dilation)
    call_from_clib(
        convolve2_nn.__name__,
        ctypes.pointer(out),
        signal,
        kernel,
        2,
        stride_c_shape.c_array,
        2,
        padding_c_shape.c_array,
        2,
        dilation_c_shape.c_array,
    )
    return out


def convolve2_sep(col_kernel: AFArray, row_kernel: AFArray, signal: AFArray, mode: ConvMode, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__convolve__sep.htm#gaeb6ba88155cf3ef29d93f97b147e372f
    """
    out = AFArray.create_null_pointer()
    call_from_clib(convolve2_sep.__name__, ctypes.pointer(out), col_kernel, row_kernel, signal, mode.value)
    return out


def convolve3(signal: AFArray, kernel: AFArray, mode: ConvMode, domain: ConvDomain, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__convolve3.htm#gab87335bb7ca6ce057811248c4c641182
    """
    out = AFArray.create_null_pointer()
    call_from_clib(convolve3.__name__, ctypes.pointer(out), signal, kernel, mode.value, domain.value)
    return out


def fft_convolve3(signal: AFArray, kernel: AFArray, mode: ConvMode, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__convolve3.htm#ga75f2ee15302b8aa7fa2ba0f5589af00e
    """
    out = AFArray.create_null_pointer()
    call_from_clib(fft_convolve3.__name__, ctypes.pointer(out), signal, kernel, mode.value)
    return out
