import ctypes

from arrayfire_wrapper.defines import AFArray, CDimT
from arrayfire_wrapper.lib._constants import Interp
from arrayfire_wrapper.lib._utility import call_from_clib


def approx1(arr: AFArray, pos: AFArray, method: Interp, off_grid: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__approx1.htm#ga10c8bfceea06e7d5dfffecb893b0ccbd
    """
    out = AFArray.create_null_pointer()
    call_from_clib(approx1.__name__, ctypes.pointer(out), arr, pos, method.value, ctypes.c_float(off_grid))
    return out


def approx1_uniform(
    arr: AFArray, pos: AFArray, interp_dim: int, idx_start: float, idx_step: float, method: Interp, off_grid: float, /
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__approx1.htm#ga888bded6f0349efd16cc15b4d7687d3c
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        approx1_uniform.__name__,
        ctypes.pointer(out),
        arr,
        pos,
        CDimT(interp_dim),
        ctypes.c_double(idx_start),
        ctypes.c_double(idx_step),
        method.value,
        ctypes.c_float(off_grid),
    )
    return out


def approx1_v2(arr: AFArray, pos: AFArray, method: Interp, off_grid: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__approx1.htm#ga24be84d2725a52cbee03eeed2a8d78d4
    """
    out = AFArray.create_null_pointer()
    call_from_clib(approx1_v2.__name__, ctypes.pointer(out), arr, pos, method.value, ctypes.c_float(off_grid))
    return out


def approx1_uniform_v2(
    arr: AFArray, pos: AFArray, interp_dim: int, idx_start: float, idx_step: float, method: Interp, off_grid: float, /
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__approx1.htm#gac3a09153cadef70a5f0f5d414c682857
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        approx1_uniform_v2.__name__,
        ctypes.pointer(out),
        arr,
        pos,
        CDimT(interp_dim),
        ctypes.c_double(idx_start),
        ctypes.c_double(idx_step),
        method.value,
        ctypes.c_float(off_grid),
    )
    return out


def approx2(arr: AFArray, pos0: AFArray, pos1: AFArray, method: Interp, off_grid: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__approx2.htm#gadd7c8ac60b5dc06999e16c7c8e555699
    """
    out = AFArray.create_null_pointer()
    call_from_clib(approx2.__name__, ctypes.pointer(out), arr, pos0, pos1, method.value, ctypes.c_float(off_grid))
    return out


def approx2_uniform(
    arr: AFArray,
    pos0: AFArray,
    interp_dim0: int,
    idx_start_dim0: float,
    idx_step_dim0: float,
    pos1: AFArray,
    interp_dim1: int,
    idx_start_dim1: float,
    idx_step_dim1: float,
    method: Interp,
    off_grid: float,
    /,
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__approx2.htm#gabad5f584e30df679b9c57193603e94b6
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        approx1_uniform.__name__,
        ctypes.pointer(out),
        arr,
        pos0,
        CDimT(interp_dim0),
        ctypes.c_double(idx_start_dim0),
        ctypes.c_double(idx_step_dim0),
        pos1,
        CDimT(interp_dim1),
        ctypes.c_double(idx_start_dim1),
        ctypes.c_double(idx_step_dim1),
        method.value,
        ctypes.c_float(off_grid),
    )
    return out


def approx2_v2(arr: AFArray, pos0: AFArray, pos1: AFArray, method: Interp, off_grid: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__approx2.htm#ga7448f8eecc9e454d814833b1b19af699
    """
    out = AFArray.create_null_pointer()
    call_from_clib(approx2_v2.__name__, ctypes.pointer(out), arr, pos0, pos1, method.value, ctypes.c_float(off_grid))
    return out


def approx2_uniform_v2(
    arr: AFArray,
    pos0: AFArray,
    interp_dim0: int,
    idx_start_dim0: float,
    idx_step_dim0: float,
    pos1: AFArray,
    interp_dim1: int,
    idx_start_dim1: float,
    idx_step_dim1: float,
    method: Interp,
    off_grid: float,
    /,
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__signal__func__approx2.htm#ga010d987451e971518b39540aa9aba550
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        approx2_uniform_v2.__name__,
        ctypes.pointer(out),
        arr,
        pos0,
        CDimT(interp_dim0),
        ctypes.c_double(idx_start_dim0),
        ctypes.c_double(idx_step_dim0),
        pos1,
        CDimT(interp_dim1),
        ctypes.c_double(idx_start_dim1),
        ctypes.c_double(idx_step_dim1),
        method.value,
        ctypes.c_float(off_grid),
    )
    return out
