import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def cholesky(arr: AFArray, is_upper: bool, /) -> tuple[AFArray, int]:
    """
    source: https://arrayfire.org/docs/group__lapack__factor__func__cholesky.htm#gab9428b008330b309bb6359a6b1c474ca
    """
    out = AFArray.create_null_pointer()
    info = ctypes.c_int(0)
    call_from_clib(cholesky.__name__, ctypes.pointer(out), ctypes.pointer(info), arr, is_upper)
    return (out, info.value)


def cholesky_inplace(arr: AFArray, is_upper: bool, /) -> bool:
    """
    source: https://arrayfire.org/docs/group__lapack__factor__func__cholesky.htm#gad87cdbff216808d53640ddb9fb71d293
    """
    info = ctypes.c_int(0)
    call_from_clib(cholesky_inplace.__name__, ctypes.pointer(info), arr, is_upper)
    return bool(info.value)


def lu(arr: AFArray, /) -> tuple[AFArray, AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__lapack__factor__func__lu.htm#gad0ddb39e6a9cf22afe24bd45fd892086
    """
    lower = AFArray.create_null_pointer()
    upper = AFArray.create_null_pointer()
    pivot = AFArray.create_null_pointer()
    call_from_clib(lu.__name__, ctypes.pointer(lower), ctypes.pointer(upper), ctypes.pointer(pivot), arr)
    return (lower, upper, pivot)


def lu_inplace(arr: AFArray, is_lapack_pivot: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__lapack__factor__func__lu.htm#ga0adcdc4b189c34644a7153c6ce9c4f7f
    """
    pivot = AFArray.create_null_pointer()
    call_from_clib(lu_inplace.__name__, ctypes.pointer(pivot), arr, is_lapack_pivot)
    return pivot


def qr(arr: AFArray, /) -> tuple[AFArray, AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__lapack__factor__func__qr.htm#ga589a9f6b4101ca354d272e9465cf7709
    """
    q = AFArray.create_null_pointer()
    r = AFArray.create_null_pointer()
    tau = AFArray.create_null_pointer()
    call_from_clib(qr.__name__, ctypes.pointer(q), ctypes.pointer(r), ctypes.pointer(tau), arr)
    return (q, r, tau)


def qr_inplace(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__lapack__factor__func__qr.htm#ga906d458fe5d4bccd9884ed26dec3c14a
    """
    tau = AFArray.create_null_pointer()
    call_from_clib(qr_inplace.__name__, ctypes.pointer(tau), arr)
    return tau


def svd(arr: AFArray, /) -> tuple[AFArray, AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__lapack__factor__func__svd.htm#gae76f30656dce9bd67246ec652983ec21
    """
    u = AFArray.create_null_pointer()
    s = AFArray.create_null_pointer()
    vt = AFArray.create_null_pointer()
    call_from_clib(svd.__name__, ctypes.pointer(u), ctypes.pointer(s), ctypes.pointer(vt), arr)
    return (u, s, vt)


def svd_inplace(arr: AFArray, /) -> tuple[AFArray, AFArray, AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__lapack__factor__func__svd.htm#ga80b31f7671bf00143dd992df8d585a2d
    """
    u = AFArray.create_null_pointer()
    s = AFArray.create_null_pointer()
    vt = AFArray.create_null_pointer()
    call_from_clib(svd.__name__, ctypes.pointer(u), ctypes.pointer(s), ctypes.pointer(vt), arr)
    return (u, s, vt, arr)
