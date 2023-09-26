import ctypes

from arrayfire_wrapper.defines import AFArray, CDimT
from arrayfire_wrapper.lib._utility import call_from_clib


def wrap(
    image: AFArray, ox: int, oy: int, wx: int, wy: int, sx: int, sy: int, px: int, py: int, is_column: bool, /
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__wrap.htm#gaace920603110e045a0e251ca8ca4377c
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        wrap.__name__,
        ctypes.pointer(out),
        image,
        CDimT(ox),
        CDimT(oy),
        CDimT(wx),
        CDimT(wy),
        CDimT(sx),
        CDimT(sy),
        CDimT(px),
        CDimT(py),
        is_column,
    )
    return out


def wrap_v2(
    image: AFArray, ox: int, oy: int, wx: int, wy: int, sx: int, sy: int, px: int, py: int, is_column: bool, /
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__wrap.htm#gaff897271aa30538fff13f60cda494d32
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        wrap_v2.__name__,
        ctypes.pointer(out),
        image,
        CDimT(ox),
        CDimT(oy),
        CDimT(wx),
        CDimT(wy),
        CDimT(sx),
        CDimT(sy),
        CDimT(px),
        CDimT(py),
        is_column,
    )
    return out


def unwrap(
    image: AFArray, ox: int, oy: int, wx: int, wy: int, sx: int, sy: int, px: int, py: int, is_column: bool, /
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__image__func__unwrap.htm#ga79b946d02b227e217097a7fece23dcde
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        unwrap.__name__,
        ctypes.pointer(out),
        image,
        CDimT(ox),
        CDimT(oy),
        CDimT(wx),
        CDimT(wy),
        CDimT(sx),
        CDimT(sy),
        CDimT(px),
        CDimT(py),
        is_column,
    )
    return out
