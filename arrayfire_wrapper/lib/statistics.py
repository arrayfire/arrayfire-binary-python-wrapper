import ctypes

from arrayfire_wrapper._backend import _backend
from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._constants import TopK, VarianceBias
from arrayfire_wrapper.lib._error_handler import safe_call


def corrcoef(x: AFArray, y: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__stat__func__corrcoef.htm#ga26b894c86731234136bfe1342453d8a7
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(_backend.clib.af_corrcoef(ctypes.pointer(real), ctypes.pointer(imag), x, y))
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def cov(x: AFArray, y: AFArray, bias: VarianceBias = VarianceBias.DEFAULT, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__stat__func__cov.htm#ga1c1c9a1d919efb02729958a91666162f
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_cov_v2(ctypes.pointer(out), x, y, bias.value))
    return out


def mean(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__stat__func__mean.htm#ga762600f4aa698a1de34ce72f7d4a0d89
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_mean(ctypes.pointer(out), arr, ctypes.c_int(dim)))
    return out


def mean_all(arr: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__stat__func__mean.htm#ga87fd44bb47e6ea6380ea8b7dadd2f4e8
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(_backend.clib.af_mean_all(ctypes.pointer(real), ctypes.pointer(imag), arr))
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def mean_all_weighted(arr: AFArray, weights: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__stat__func__mean.htm#ga008221f09b128799b2382493916a4bc8
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(_backend.clib.af_mean_all_weighted(ctypes.pointer(real), ctypes.pointer(imag), arr, weights))
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def mean_weighted(arr: AFArray, weights: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__stat__func__mean.htm#ga008221f09b128799b2382493916a4bc8
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_mean_weighted(ctypes.pointer(out), arr, weights, ctypes.c_int(dim)))
    return out


def median(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__stat__func__median.htm#ga79ced0d340f8cbebb9fd1e0e75b7ee9e
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_median(ctypes.pointer(out), arr, ctypes.c_int(dim)))
    return out


def median_all(arr: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__stat__func__median.htm#ga82b3d518bb7838eb6795af4a92c08b92
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(_backend.clib.af_median_all(ctypes.pointer(real), ctypes.pointer(imag), arr))
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def stdev(arr: AFArray, dim: int, bias: VarianceBias = VarianceBias.DEFAULT, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__stat__func__stdev.htm#gac96bb45869add8c949020112b5350ea5
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_stdev_v2(ctypes.pointer(out), arr, bias.value, ctypes.c_int(dim)))
    return out


def stdev_all(arr: AFArray, bias: VarianceBias = VarianceBias.DEFAULT, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__stat__func__stdev.htm#ga1642775a5abef993213306e0fd6d72cf
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(_backend.clib.f_stdev_all_v2(ctypes.pointer(real), ctypes.pointer(imag), arr, bias.value))
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def topk(arr: AFArray, k: int, dim: int = 0, order: TopK = TopK.DEFAULT, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__stat__func__topk.htm#gaea5a2a6aa11ce7d25f7a7c0ffadbdf51
    """
    values = AFArray.create_null_pointer()
    indices = AFArray.create_null_pointer()

    safe_call(
        _backend.clib.af_topk(ctypes.pointer(values), ctypes.pointer(indices), arr, k, ctypes.c_int(dim), order.value)
    )

    return values, indices


def var(arr: AFArray, dim: int, bias: VarianceBias = VarianceBias.DEFAULT, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__stat__func__var.htm#ga7782e8de146ef2e7816aa75448ef8648
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_var_v2(ctypes.pointer(out), arr, bias.value, ctypes.c_int(dim)))
    return out


def var_all(arr: AFArray, bias: VarianceBias = VarianceBias.DEFAULT, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__stat__func__var.htm#gad6a3ed0dd2e0b6878eb6538211c55f5b
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(_backend.clib.af_var_all_v2(ctypes.pointer(real), ctypes.pointer(imag), arr, bias.value))
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def var_all_weighted(arr: AFArray, weights: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__stat__func__var.htm#ga26f83014829926e3112de3435a87ac1d
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    safe_call(_backend.clib.af_var_all_weighted(ctypes.pointer(real), ctypes.pointer(imag), arr, weights))
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def var_weighted(arr: AFArray, weights: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__stat__func__var.htm#ga06ad132cb12a5760c2058278456d041e
    """
    out = AFArray.create_null_pointer()
    safe_call(_backend.clib.af_var_weighted(ctypes.pointer(out), arr, weights, ctypes.c_int(dim)))
    return out
