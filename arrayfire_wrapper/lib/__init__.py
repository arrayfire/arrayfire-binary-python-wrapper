# flake8: noqa

# Functions to Create and Modify Arrays

__all__ = ["assign_gen", "assign_seq"]

from .create_and_modify_array.assignment_and_indexing.assign import assign_gen, assign_seq

__all__ += ["CIndexStructure", "IndexStructure", "ParallelRange", "get_indices"]

from .create_and_modify_array.assignment_and_indexing.indexing import (
    CIndexStructure,
    IndexStructure,
    ParallelRange,
    get_indices,
)

__all__ += ["lookup"]

from .create_and_modify_array.assignment_and_indexing.lookup import lookup

__all__ += [
    "constant",
    "constant_complex",
    "constant_long",
    "constant_ulong",
]
from .create_and_modify_array.create_array.constant import constant, constant_complex, constant_long, constant_ulong

__all__ += ["diag_create", "diag_extract"]

from .create_and_modify_array.create_array.diag import diag_create, diag_extract

__all__ += ["identity"]

from .create_and_modify_array.create_array.identity import identity

__all__ += ["iota"]

from .create_and_modify_array.create_array.iota import iota

__all__ += ["lower"]

from .create_and_modify_array.create_array.lower import lower

__all__ += ["pad"]

from .create_and_modify_array.create_array.pad import pad

__all__ += [
    "AFRandomEngineHandle",
    "create_random_engine",
    "random_engine_get_seed",
    "random_engine_get_type",
    "random_engine_set_seed",
    "random_engine_set_type",
    "random_uniform",
    "randu",
    "release_random_engine",
]

from .create_and_modify_array.create_array.random_number_generation import (
    AFRandomEngineHandle,
    create_random_engine,
    random_engine_get_seed,
    random_engine_get_type,
    random_engine_set_seed,
    random_engine_set_type,
    random_uniform,
    randu,
    release_random_engine,
)

__all__ += ["range"]

from .create_and_modify_array.create_array.range import range

__all__ += ["upper"]

from .create_and_modify_array.create_array.upper import upper

__all__ += ["cast", "isinf", "isnan", "iszero"]

from .create_and_modify_array.helper_functions import cast, isinf, isnan, iszero

__all__ += [
    "copy_array",
    "create_array",
    "create_handle",
    "device_array",
    "eval",
    "eval_multiple",
    "get_data_ptr",
    "get_data_ref_count",
    "get_dims",
    "get_elements",
    "get_manual_eval_flag",
    "get_numdims",
    "get_scalar",
    "get_type",
    "is_bool",
    "is_column",
    "is_complex",
    "is_double",
    "is_empty",
    "is_floating",
    "is_half",
    "is_integer",
    "is_real",
    "is_realfloating",
    "is_row",
    "is_scalar",
    "is_single",
    "is_sparse",
    "is_vector",
    "release_array",
    "retain_array",
    "set_manual_eval_flag",
    "write_array",
]

from .create_and_modify_array.manage_array import (
    copy_array,
    create_array,
    create_handle,
    device_array,
    eval,
    eval_multiple,
    get_data_ptr,
    get_data_ref_count,
    get_dims,
    get_elements,
    get_manual_eval_flag,
    get_numdims,
    get_scalar,
    get_type,
    is_bool,
    is_column,
    is_complex,
    is_double,
    is_empty,
    is_floating,
    is_half,
    is_integer,
    is_real,
    is_realfloating,
    is_row,
    is_scalar,
    is_single,
    is_sparse,
    is_vector,
    release_array,
    retain_array,
    set_manual_eval_flag,
    write_array,
)

__all__ += [
    "alloc_device",
    "alloc_host",
    "alloc_pinned",
    "device_gc",
    "device_info",
    "device_mem_info",
    "free_device",
    "free_host",
    "free_pinned",
    "get_dbl_support",
    "get_device",
    "get_device_count",
    "get_device_ptr",
    "get_half_support",
    "get_kernel_cache_directory",
    "get_mem_step_size",
    "info",
    "info_string",
    "init",
    "is_locked_array",
    "lock_array",
    "lock_device_ptr",
    "print_mem_info",
    "set_device",
    "set_kernel_cache_directory",
    "set_mem_step_size",
    "unlock_array",
    "unlock_device_ptr",
]

from .create_and_modify_array.manage_device import (
    alloc_device,
    alloc_host,
    alloc_pinned,
    device_gc,
    device_info,
    device_mem_info,
    free_device,
    free_host,
    free_pinned,
    get_dbl_support,
    get_device,
    get_device_count,
    get_device_ptr,
    get_half_support,
    get_kernel_cache_directory,
    get_mem_step_size,
    info,
    info_string,
    init,
    is_locked_array,
    lock_array,
    lock_device_ptr,
    print_mem_info,
    set_device,
    set_kernel_cache_directory,
    set_mem_step_size,
    unlock_array,
    unlock_device_ptr,
)

__all__ += [
    "flat",
    "flip",
    "join",
    "join_many",
    "moddims",
    "reorder",
    "replace",
    "replace_scalar",
    "select",
    "select_scalar_l",
    "select_scalar_r",
    "shift",
    "tile",
    "transpose",
    "transpose_inplace",
]

from .create_and_modify_array.move_and_reorder import (
    flat,
    flip,
    join,
    join_many,
    moddims,
    reorder,
    replace,
    replace_scalar,
    select,
    select_scalar_l,
    select_scalar_r,
    shift,
    tile,
    transpose,
    transpose_inplace,
)

# Interface Functions

__all__ += [
    "cublas_set_math_mode",
    "get_native_id",
    "get_stream",
    "set_native_id",
]

from .interface_functions.cuda import cublas_set_math_mode, get_native_id, get_stream, set_native_id

# Mathematical Functions

__all__ += ["add", "bitshiftl", "bitshiftr", "div", "mul", "sub"]

from .mathematical_functions.arithmetic_operations import add, bitshiftl, bitshiftr, div, mul, sub

__all__ += ["conjg", "cplx", "cplx2", "imag", "real"]

from .mathematical_functions.complex_operations import conjg, cplx, cplx2, imag, real

__all__ += [
    "cbrt",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "factorial",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "pow",
    "pow2",
    "root",
    "rsqrt",
    "sqrt",
    "tgamma",
]

from .mathematical_functions.exp_and_log_functions import (
    cbrt,
    erf,
    erfc,
    exp,
    expm1,
    factorial,
    lgamma,
    log,
    log1p,
    log2,
    log10,
    pow,
    pow2,
    root,
    rsqrt,
    sqrt,
    tgamma,
)

__all__ += ["acosh", "asinh", "atanh", "cosh", "sinh", "tanh"]

from .mathematical_functions.hyperbolic_functions import acosh, asinh, atanh, cosh, sinh, tanh

__all__ += [
    "and_",
    "bitand",
    "bitnot",
    "bitor",
    "bitxor",
    "eq",
    "ge",
    "gt",
    "le",
    "lt",
    "neq",
    "not_",
    "or_",
]

from .mathematical_functions.logical_operations import (
    and_,
    bitand,
    bitnot,
    bitor,
    bitxor,
    eq,
    ge,
    gt,
    le,
    lt,
    neq,
    not_,
    or_,
)

__all__ += [
    "abs_",
    "arg",
    "ceil",
    "clamp",
    "floor",
    "hypot",
    "max_",
    "min_",
    "mod",
    "neg",
    "rem",
    "round_",
    "sign",
    "trunc",
]

from .mathematical_functions.numeric_functions import (
    abs_,
    arg,
    ceil,
    clamp,
    floor,
    hypot,
    max_,
    min_,
    mod,
    neg,
    rem,
    round_,
    sign,
    trunc,
)

__all__ += ["acos", "asin", "atan", "atan2", "cos", "sin", "tan"]

from .mathematical_functions.trigonometric_functions import acos, asin, atan, atan2, cos, sin, tan

# Functions to Work with Internal Array Layout

__all__ += [
    "create_strided_array",
    "get_offset",
    "get_strides",
    "is_linear",
    "is_owner",
]

from .array_layout import create_strided_array, get_offset, get_strides, is_linear, is_owner

# Statistics

__all__ += [
    "TopK",
    "VarianceBias",
    "corrcoef",
    "cov",
    "mean",
    "mean_all",
    "mean_all_weighted",
    "mean_weighted",
    "median",
    "median_all",
    "stdev",
    "stdev_all",
    "topk",
    "var",
    "var_all",
    "var_all_weighted",
    "var_weighted",
]

from .statistics import (
    TopK,
    VarianceBias,
    corrcoef,
    cov,
    mean,
    mean_all,
    mean_all_weighted,
    mean_weighted,
    median,
    median_all,
    stdev,
    stdev_all,
    topk,
    var,
    var_all,
    var_all_weighted,
    var_weighted,
)
