import ctypes

from arrayfire_wrapper.defines import AFArray, CShape
from arrayfire_wrapper.lib._constants import Pad
from arrayfire_wrapper.lib._utility import call_from_clib


def pad(arr: AFArray, begin_shape: tuple[int, ...], end_shape: tuple[int, ...], border_type: Pad) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__pad.htm#gabe75c767e4e89f82a4e5864bf6e1ef15
    """
    out = AFArray.create_null_pointer()
    begin_c_shape = CShape(*begin_shape)
    end_c_shape = CShape(*end_shape)
    call_from_clib(
        pad.__name__,
        ctypes.pointer(out),
        arr,
        4,
        begin_c_shape.c_array,
        4,
        end_c_shape.c_array,
        border_type.value,
    )
    return NotImplemented
