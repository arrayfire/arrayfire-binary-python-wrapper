import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import call_from_clib


def all_true(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__reduce__func__all__true.htm#ga068708be5177a0aa3788af140bb5ebd6
    """
    out = AFArray.create_null_pointer()
    call_from_clib(all_true.__name__, ctypes.pointer(out), arr, dim)
    return out


def all_true_all(arr: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__all__true.htm#ga068708be5177a0aa3788af140bb5ebd6
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(all_true_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), arr)
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def all_true_by_key(keys: AFArray, values: AFArray, dim: int, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/algorithm_8h.htm#a65fa5577c81a2c2fcf7406bf48cc014a
    """
    out_keys = AFArray.create_null_pointer()
    out_values = AFArray.create_null_pointer()
    call_from_clib(
        all_true_by_key.__name__, ctypes.pointer(out_keys), ctypes.pointer(out_values), keys, values, ctypes.c_int(dim)
    )
    return (out_keys, out_values)


def any_true(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__reduce__func__any__true.htm#ga7c275cda2cfc8eb0bd20ea86472ca0d5
    """
    out = AFArray.create_null_pointer()
    call_from_clib(any_true.__name__, ctypes.pointer(out), arr, dim)
    return out


def any_true_all(arr: AFArray, /) -> int | float | bool | complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__any__true.htm#ga47d991276bb5bf8cdba8340e8751e536
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(any_true_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), arr)
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def any_true_by_key(keys: AFArray, values: AFArray, dim: int, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__reduce__func__anytrue__by__key.htm#ga973fd650f8a57533f675cfd7ad6f0718
    """
    out_keys = AFArray.create_null_pointer()
    out_values = AFArray.create_null_pointer()
    call_from_clib(
        any_true_by_key.__name__, ctypes.pointer(out_keys), ctypes.pointer(out_values), keys, values, ctypes.c_int(dim)
    )
    return (out_keys, out_values)


def count(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__reduce__func__count.htm#gaf2664c25ee6ca30aa3f5aa77db789f95
    """
    out = AFArray.create_null_pointer()
    call_from_clib(count.__name__, ctypes.pointer(out), arr, dim)
    return out


def count_all(arr: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__count.htm#ga38699c5ce172c15e9850a9eda6050da5
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(count_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), arr)
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def count_by_key(keys: AFArray, values: AFArray, dim: int, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__reduce__func__count__by__key.htm#ga96b01fd7375b3a3cb065ba860885e723
    """
    out_keys = AFArray.create_null_pointer()
    out_values = AFArray.create_null_pointer()
    call_from_clib(
        count_by_key.__name__, ctypes.pointer(out_keys), ctypes.pointer(out_values), keys, values, ctypes.c_int(dim)
    )
    return (out_keys, out_values)


def imax(arr: AFArray, dim: int, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__reduce__func__max.htm#gaf0e6a523e2e435d5409d5d8cb843d8a2
    """
    out = AFArray.create_null_pointer()
    out_idx = AFArray.create_null_pointer()
    call_from_clib(imax.__name__, ctypes.pointer(out), ctypes.pointer(out_idx), arr, ctypes.c_int(dim))
    return (out, out_idx)


def imax_all(arr: AFArray, /) -> tuple[complex, int]:
    """
    source: https://arrayfire.org/docs/group__reduce__func__max.htm#gaea009bd51145be2fcc688b2390725401
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    out_idx = ctypes.c_uint(0)
    call_from_clib(imax_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), ctypes.pointer(out_idx), arr)
    complex_value = real.value if imag.value == 0 else real.value + imag.value * 1j
    return (complex_value, out_idx.value)


def max(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__reduce__func__max.htm#ga267f32b8dbb1b508e8738e3748d8dc3f
    """
    out = AFArray.create_null_pointer()
    call_from_clib(max.__name__, ctypes.pointer(out), arr, dim)
    return out


def max_all(arr: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__max.htm#ga5f71ab6056943723149585d2aebade7c
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(max_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), arr)
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def max_ragged(arr: AFArray, ragged_len: AFArray, dim: int, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__reduce__func__max.htm#ga564bbeca8e4c243355979a6cb5dc4970
    """
    out_values = AFArray.create_null_pointer()
    out_idx = AFArray.create_null_pointer()
    call_from_clib(max_ragged.__name__, ctypes.pointer(out_values), ctypes.pointer(out_idx), arr, ragged_len, dim)
    return (out_values, out_idx)


def max_by_key(keys: AFArray, values: AFArray, dim: int, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__reduce__func__max__by__key.htm#ga002d03c0ebd674644c8a6831ebb775e2
    """
    out_keys = AFArray.create_null_pointer()
    out_values = AFArray.create_null_pointer()
    call_from_clib(max_by_key.__name__, ctypes.pointer(out_keys), ctypes.pointer(out_values), keys, values, dim)
    return (out_keys, out_values)


def imin(arr: AFArray, dim: int, /) -> tuple[AFArray, AFArray]:
    """
    source: https://arrayfire.org/docs/group__reduce__func__min.htm#ga2f65943090e0c2317bd682c25594b901
    """
    out = AFArray.create_null_pointer()
    out_idx = AFArray.create_null_pointer()
    call_from_clib(imin.__name__, ctypes.pointer(out), ctypes.pointer(out_idx), arr, ctypes.c_int(dim))
    return (out, out_idx)


def imin_all(arr: AFArray, /) -> tuple[complex, int]:
    """
    source: https://arrayfire.org/docs/group__reduce__func__min.htm#gae75785af0fdfcbb1f4c34461235f5206
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    out_idx = ctypes.c_uint(0)
    call_from_clib(imin_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), ctypes.pointer(out_idx), arr)
    complex_value = real.value if imag.value == 0 else real.value + imag.value * 1j
    return (complex_value, out_idx.value)


def min(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__reduce__func__min.htm#ga2ac4c8d9ba613dbc9bfec0bee7be8eb8
    """
    out = AFArray.create_null_pointer()
    call_from_clib(min.__name__, ctypes.pointer(out), arr, dim)
    return out


def min_all(arr: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__min.htm#gab10198ae7ead1dc10f220d576f118104
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(min_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), arr)
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def product(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__reduce__func__product.htm#ga2be338d39be30ad22dddf658a4f5676e
    """
    out = AFArray.create_null_pointer()
    call_from_clib(product.__name__, ctypes.pointer(out), arr, dim)
    return out


def product_all(arr: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__product.htm#gad226a6ec77c12fd16cf42e3fe3264e22
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(product_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), arr)
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def product_nan(arr: AFArray, dim: int, nan_value: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__reduce__func__product.htm#ga1d25447c16d492767ba7efa7ee72a36e
    """
    out = AFArray.create_null_pointer()
    call_from_clib(product_nan.__name__, ctypes.pointer(out), arr, dim, ctypes.c_double(nan_value))
    return out


def product_nan_all(arr: AFArray, nan_value: float, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__product.htm#gaca78d54c53a33b419bfdb5c64accbc7b
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(
        product_nan_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), arr, ctypes.c_double(nan_value)
    )
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def sum(arr: AFArray, dim: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__reduce__func__sum.htm#gacd4917c2e916870ebdf54afc2f61d533
    """
    out = AFArray.create_null_pointer()
    call_from_clib(sum.__name__, ctypes.pointer(out), arr, dim)
    return out


def sum_all(arr: AFArray, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__sum.htm#gabc009d04df0faf29ba1e381c7badde58
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(sum_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), arr)
    return real.value if imag.value == 0 else real.value + imag.value * 1j


def sum_nan(arr: AFArray, dim: int, nan_value: float, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__reduce__func__sum.htm#ga52461231e2d9995f689b7f23eea0e798
    """
    out = AFArray.create_null_pointer()
    call_from_clib(sum_nan.__name__, ctypes.pointer(out), arr, dim, ctypes.c_double(nan_value))
    return out


def sum_nan_all(arr: AFArray, nan_value: float, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__reduce__func__sum.htm#gabc009d04df0faf29ba1e381c7badde58
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(sum_nan_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), arr, ctypes.c_double(nan_value))
    return real.value if imag.value == 0 else real.value + imag.value * 1j
