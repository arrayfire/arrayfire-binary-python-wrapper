import ctypes

from arrayfire_wrapper.defines import AFArray, CShape
from arrayfire_wrapper.lib._constants import ConvGradient
from arrayfire_wrapper.lib._utility import call_from_clib


def convolve2_gradient_nn(
    incoming_gradient: AFArray,
    original_signal: AFArray,
    original_filter: AFArray,
    convolved_output: AFArray,
    strides: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    grad_type: ConvGradient,
    /,
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__ml__convolution.htm#ga3dc8cbebcec76e5c1804ff377b4e1cfd
    """
    out = AFArray.create_null_pointer()
    c_strides = CShape(*strides)
    c_padding = CShape(*padding)
    c_dilation = CShape(*dilation)

    call_from_clib(
        convolve2_gradient_nn.__name__,
        ctypes.pointer(out),
        incoming_gradient,
        original_signal,
        original_filter,
        convolved_output,
        len(strides),
        c_strides.c_array,
        len(padding),
        c_padding.c_array,
        len(dilation),
        c_dilation.c_array,
        grad_type.value,
    )
    return out
