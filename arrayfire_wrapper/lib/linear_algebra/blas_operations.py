import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._constants import MatProp
from arrayfire_wrapper.lib._utility import call_from_clib


def dot(lhs: AFArray, rhs: AFArray, lhs_opts: MatProp, rhs_opts: MatProp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__blas__func__dot.htm#ga030ea5d9b694a4d3847f69254ab4a90d
    """
    out = AFArray.create_null_pointer()
    call_from_clib(dot.__name__, ctypes.pointer(out), lhs, rhs, lhs_opts.value, rhs_opts.value)
    return out


def dot_all(lhs: AFArray, rhs: AFArray, lhs_opts: MatProp, rhs_opts: MatProp, /) -> complex:
    """
    source: https://arrayfire.org/docs/group__blas__func__dot.htm#gafb619ba32e85dfac62237929da911995
    """
    real = ctypes.c_double(0)
    imag = ctypes.c_double(0)
    call_from_clib(
        dot_all.__name__, ctypes.pointer(real), ctypes.pointer(imag), lhs, rhs, lhs_opts.value, rhs_opts.value
    )
    return real.value if imag.value == 0 else real.value + imag.value * 1j


# FIXME
# def gemm(
#     lhs: AFArray,
#     rhs: AFArray,
#     alpha_ptr: ctypes._Pointer,
#     beta_ptr: ctypes._Pointer,
#     lhs_opts: MatProp,
#     rhs_opts: MatProp,
#     res: None | AFArray = None,
#     /,
# ) -> None | AFArray:
#     """
#     source: https://arrayfire.org/docs/group__blas__func__matmul.htm#ga0463ae584163128718237b02faf5caf7
#     """
#     out = AFArray.create_null_pointer() if not res else res
#     call_from_clib(
#         gemm.__name__, ctypes.pointer(out), lhs_opts.value, rhs_opts.value, ctypes.c_void_p, alpha_ptr, beta_ptr
#     )


def matmul(lhs: AFArray, rhs: AFArray, lhs_opts: MatProp, rhs_opts: MatProp, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__blas__func__matmul.htm#ga3f3f29358b44286d19ff2037547649fe
    """
    out = AFArray.create_null_pointer()
    call_from_clib(matmul.__name__, ctypes.pointer(out), lhs, rhs, lhs_opts.value, rhs_opts.value)
    return out
