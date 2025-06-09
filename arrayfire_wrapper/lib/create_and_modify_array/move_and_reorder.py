import ctypes

from arrayfire_wrapper.defines import AFArray, CShape
from arrayfire_wrapper.lib._utility import call_from_clib


def flat(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__flat.htm#gac6dfb22cbd3b151ddffb9a4ddf74455e
    """
    out = AFArray.create_null_pointer()
    call_from_clib(flat.__name__, ctypes.pointer(out), arr)
    return out


def flip(arr: AFArray, dim: int = 0, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__flip.htm#gac0795e2a4343ea8f897b3b7d23802ccb
    """
    out = AFArray.create_null_pointer()
    call_from_clib(flip.__name__, ctypes.pointer(out), arr, ctypes.c_int(dim))
    return out


def join(dim: int, first: AFArray, second: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__join.htm#ga4c0b185d13b49023cc22c0269eedbdb2
    """
    out = AFArray.create_null_pointer()
    call_from_clib(join.__name__, ctypes.pointer(out), dim, first, second)
    return out


def join_many(dim: int, n_arrays: int, *inputs: AFArray) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__join.htm#ga67a36384247f6bb40254e0cb2e6d5d5c
    """
    out = AFArray.create_null_pointer()

    size = 10 if len(inputs) > 10 else len(inputs)  # NOTE: API limitations
    inputs_arr = ctypes.c_void_p * size
    inputs_arr_vector = inputs_arr(*inputs[:size])

    call_from_clib(join_many.__name__, ctypes.pointer(out), dim, n_arrays, ctypes.pointer(inputs_arr_vector))
    return out


def moddims(arr: AFArray, shape: tuple[int, ...], /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__moddims.htm#ga50442cfa497c34054c3dc4404e92667a
    """
    out = AFArray.create_null_pointer()
    c_shape = CShape(*shape)
    call_from_clib(moddims.__name__, ctypes.pointer(out), arr, len(c_shape), c_shape.c_array)
    return out


def reorder(arr: AFArray, /, d0: int = 1, d1: int = 0, d2: int = 2, d3: int = 3) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__reorder.htm#ga57383f4d00a3a86eab08dddd52c3ad3d
    """
    out = AFArray.create_null_pointer()
    call_from_clib(reorder.__name__, ctypes.pointer(out), arr, d0, d1, d2, d3)
    return out


def replace(lhs: AFArray, cond_arr: AFArray, rhs: AFArray, /) -> None:
    """
    source: https://arrayfire.org/docs/group__data__func__replace.htm#ga6d285cd28d8c380fbc0dafd5296703b5
    """
    call_from_clib(replace.__name__, lhs, cond_arr, rhs)


def replace_scalar(lhs: AFArray, cond_arr: AFArray, rhs: int | float, /) -> None:
    """
    source: https://arrayfire.org/docs/group__data__func__replace.htm#ga58449937228761176de47b1d75d689d8
    """
    call_from_clib(replace.__name__, lhs, cond_arr, ctypes.c_double(rhs))


def select(lhs: AFArray, cond_arr: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__select.htm#gac4af16e31ddd5ddcf09b670f676fd093
    """
    out = AFArray.create_null_pointer()
    call_from_clib(select.__name__, ctypes.pointer(out), cond_arr, lhs, rhs)
    return out


def select_scalar_l(lhs: int | float, cond_arr: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__select.htm#gac4af16e31ddd5ddcf09b670f676fd093
    """
    out = AFArray.create_null_pointer()
    call_from_clib(select_scalar_l.__name__, ctypes.pointer(out), cond_arr, ctypes.c_double(lhs), rhs)
    return out


def select_scalar_r(lhs: AFArray, cond_arr: AFArray, rhs: int | float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__select.htm#gac4af16e31ddd5ddcf09b670f676fd093
    """
    out = AFArray.create_null_pointer()
    call_from_clib(select_scalar_r.__name__, ctypes.pointer(out), cond_arr, lhs, ctypes.c_double(rhs))
    return out


def shift(arr: AFArray, /, d0: int, d1: int = 0, d2: int = 0, d3: int = 0) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__shift.htm#ga64a0cd7680b71e87f3ab372876153b66
    """
    out = AFArray.create_null_pointer()
    call_from_clib(shift.__name__, ctypes.pointer(out), arr, d0, d1, d2, d3)
    return out


def tile(arr: AFArray, /, d0: int, d1: int = 1, d2: int = 1, d3: int = 1) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__tile.htm#ga3540329723c9876839e0c790075ab076
    """
    out = AFArray.create_null_pointer()
    call_from_clib(tile.__name__, ctypes.pointer(out), arr, d0, d1, d2, d3)
    return out


def transpose(arr: AFArray, conjugate: bool, /) -> AFArray:
    """
    https://arrayfire.org/docs/group__blas__func__transpose.htm#ga716b2b9bf190c8f8d0970aef2b57d8e7
    """
    out = AFArray.create_null_pointer()
    call_from_clib(transpose.__name__, ctypes.pointer(out), arr, conjugate)
    return out


def transpose_inplace(arr: AFArray, conjugate: bool, /) -> None:
    """
    https://arrayfire.org/docs/group__blas__func__transpose.htm#ga716b2b9bf190c8f8d0970aef2b57d8e7
    """
    call_from_clib(transpose.__name__, arr, conjugate)
