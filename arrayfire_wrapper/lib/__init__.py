# flake8: noqa

# Computer Vision

__all__ = ["gloh", "orb", "sift"]

from .computer_vision.feature_descriptors import gloh, orb, sift

__all__ += ["dog", "fast", "harris", "susan"]

from .computer_vision.feature_detector import dog, fast, harris, susan

__all__ += ["hamming_matcher", "nearest_neighbour"]

from .computer_vision.feature_matchers import hamming_matcher, nearest_neighbour

__all__ += ["match_template"]

from .computer_vision.template_matching import match_template

# Functions to Create and Modify Arrays

__all__ += ["assign_gen", "assign_seq"]

from .create_and_modify_array.assignment_and_indexing.assign import assign_gen, assign_seq

__all__ += ["CIndexStructure", "IndexStructure", "ParallelRange", "get_indices"]

from .create_and_modify_array.assignment_and_indexing._indexing import (
    CIndexStructure,
    IndexStructure,
    ParallelRange,
    get_indices,
)

__all__ += ["index_gen"]

from .create_and_modify_array.assignment_and_indexing.index import index_gen

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

__all__ += ["array_to_string", "cast", "isinf", "isnan", "iszero"]

from .create_and_modify_array.helper_functions import array_to_string, cast, isinf, isnan, iszero

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
    "sync",
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
    sync,
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

# Image Processing

__all__ += [
    "CSpace",
    "YCCStd",
    "color_space",
    "gray2rgb",
    "hsv2rgb",
    "rgb2gray",
    "rgb2hsv",
    "rgb2ycbcr",
    "ycbcr2rgb",
    "CannyThreshold",
    "Diffusion",
    "Flux",
    "IterativeDeconv",
    "Pad",
    "anisotropic_diffusion",
    "bilateral",
    "canny",
    "inverse_deconv",
    "iterative_deconv",
    "maxfilt",
    "mean_shift",
    "medfilt",
    "medfilt1",
    "medfilt2",
    "minfilt",
    "sat",
    "sobel_operator",
    "gaussian_kernel",
    "hist_equal",
    "histogram",
    "Moment",
    "moments",
    "moments_all",
    "Interp",
    "resize",
    "rotate",
    "scale",
    "skew",
    "transform",
    "transform_coordinates",
    "transform_v2",
    "translate",
    "Connectivity",
    "confidence_cc",
    "regions",
    "dilate",
    "dilate3",
    "erode",
    "erode3",
    "unwrap",
    "wrap",
    "wrap_v2",
]

from .image_processing import (
    CannyThreshold,
    Connectivity,
    CSpace,
    Diffusion,
    Flux,
    Interp,
    IterativeDeconv,
    Moment,
    Pad,
    YCCStd,
    anisotropic_diffusion,
    bilateral,
    canny,
    color_space,
    confidence_cc,
    dilate,
    dilate3,
    erode,
    erode3,
    gaussian_kernel,
    gray2rgb,
    hist_equal,
    histogram,
    hsv2rgb,
    inverse_deconv,
    iterative_deconv,
    maxfilt,
    mean_shift,
    medfilt,
    medfilt1,
    medfilt2,
    minfilt,
    moments,
    moments_all,
    regions,
    resize,
    rgb2gray,
    rgb2hsv,
    rgb2ycbcr,
    rotate,
    sat,
    scale,
    skew,
    sobel_operator,
    transform,
    transform_coordinates,
    transform_v2,
    translate,
    unwrap,
    wrap,
    wrap_v2,
    ycbcr2rgb,
)

# Input and Output Functions

__all__ += [
    "delete_image_memory",
    "is_image_io_available",
    "load_image",
    "load_image_memory",
    "load_image_native",
    "read_array_index",
    "read_array_key",
    "read_array_key_check",
    "save_array",
    "save_image",
    "save_image_memory",
    "save_image_native",
]

from .input_and_output import (
    delete_image_memory,
    is_image_io_available,
    load_image,
    load_image_memory,
    load_image_native,
    read_array_index,
    read_array_key,
    read_array_key_check,
    save_array,
    save_image,
    save_image_memory,
    save_image_native,
)

# Interface Functions

__all__ += [
    "cublas_set_math_mode",
    "get_native_id",
    "get_stream",
    "set_native_id",
]

from .interface_functions.cuda import cublas_set_math_mode, get_native_id, get_stream, set_native_id

# Linear Algebra

__all__ += [
    "dot",
    "dot_all",
    "matmul",
    "is_lapack_available",
    "cholesky",
    "cholesky_inplace",
    "gemm",
    "lu",
    "lu_inplace",
    "qr",
    "qr_inplace",
    "svd",
    "svd_inplace",
    "Norm",
    "det",
    "inverse",
    "norm",
    "pinverse",
    "rank",
    "solve",
    "solve_lu",
    "Storage",
    "create_sparse_array",
    "create_sparse_array_from_dense",
    "create_sparse_array_from_ptr",
    "sparse_convert_to",
    "sparse_get_col_idx",
    "sparse_get_info",
    "sparse_get_nnz",
    "sparse_get_row_idx",
    "sparse_get_storage",
    "sparse_get_values",
    "sparse_to_dense",
]

from .linear_algebra import (
    Norm,
    Storage,
    cholesky,
    cholesky_inplace,
    create_sparse_array,
    create_sparse_array_from_dense,
    create_sparse_array_from_ptr,
    det,
    dot,
    dot_all,
    gemm,
    inverse,
    is_lapack_available,
    lu,
    lu_inplace,
    matmul,
    norm,
    pinverse,
    qr,
    qr_inplace,
    rank,
    solve,
    solve_lu,
    sparse_convert_to,
    sparse_get_col_idx,
    sparse_get_info,
    sparse_get_nnz,
    sparse_get_row_idx,
    sparse_get_storage,
    sparse_get_values,
    sparse_to_dense,
    svd,
    svd_inplace,
)

# Machine Learning

__all__ += ["ConvGradient", "convolve2_gradient_nn"]

from .machine_learning.convolutions import ConvGradient, convolve2_gradient_nn

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
    "maxof",
    "minof",
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
    maxof,
    minof,
    mod,
    neg,
    rem,
    round_,
    sign,
    trunc,
)

__all__ += ["acos", "asin", "atan", "atan2", "cos", "sin", "tan"]

from .mathematical_functions.trigonometric_functions import acos, asin, atan, atan2, cos, sin, tan

# Vector Algorithms

__all__ += [
    "BinaryOperator",
    "accum",
    "scan",
    "scan_by_key",
    "where",
]

from .vector_algorithms.inclusive_scan_operations import BinaryOperator, accum, scan, scan_by_key, where

__all__ += [
    "diff1",
    "diff2",
    "gradient",
]

from .vector_algorithms.numerical_differentiation import diff1, diff2, gradient

__all__ += [
    "all_true",
    "all_true_all",
    "all_true_by_key",
    "any_true",
    "any_true_all",
    "any_true_by_key",
    "count",
    "count_all",
    "count_by_key",
    "imax",
    "imax_all",
    "imin",
    "imin_all",
    "max",
    "max_all",
    "max_by_key",
    "max_ragged",
    "min",
    "min_all",
    "product",
    "product_all",
    "product_nan",
    "product_nan_all",
    "sum",
    "sum_all",
    "sum_nan",
    "sum_nan_all",
]

from .vector_algorithms.reduction_operations import (
    all_true,
    all_true_all,
    all_true_by_key,
    any_true,
    any_true_all,
    any_true_by_key,
    count,
    count_all,
    count_by_key,
    imax,
    imax_all,
    imin,
    imin_all,
    max,
    max_all,
    max_by_key,
    max_ragged,
    min,
    min_all,
    product,
    product_all,
    product_nan,
    product_nan_all,
    sum,
    sum_all,
    sum_nan,
    sum_nan_all,
)

__all__ += [
    "set_intersect",
    "set_union",
    "set_unique",
]

from .vector_algorithms.set_operations import set_intersect, set_union, set_unique

__all__ += [
    "sort",
    "sort_by_key",
    "sort_index",
]

from .vector_algorithms.sort_operations import sort, sort_by_key, sort_index

# Functions to Work with Internal Array Layout

__all__ += [
    "create_strided_array",
    "get_offset",
    "get_strides",
    "is_linear",
    "is_owner",
]

from .array_layout import create_strided_array, get_offset, get_strides, is_linear, is_owner

# Features

__all__ += [
    "AFFeatures",
    "create_features",
    "get_features_num",
    "get_features_orientation",
    "get_features_score",
    "get_features_size",
    "get_features_xpos",
    "get_features_ypos",
    "release_features",
    "retain_features",
]
from .features import (
    AFFeatures,
    create_features,
    get_features_num,
    get_features_orientation,
    get_features_score,
    get_features_size,
    get_features_xpos,
    get_features_ypos,
    release_features,
    retain_features,
)

# Statistics

__all__ += [
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

# Signal Processing

__all__ += [
    "convolve1",
    "convolve2",
    "convolve2_nn",
    "convolve2_sep",
    "convolve3",
    "fft_convolve1",
    "fft_convolve2",
    "fft_convolve3",
]

from .signal_processing.convolutions import (
    convolve1,
    convolve2,
    convolve2_nn,
    convolve2_sep,
    convolve3,
    fft_convolve1,
    fft_convolve2,
    fft_convolve3,
)

__all__ += [
    "fft",
    "fft2",
    "fft2_c2r",
    "fft2_inplace",
    "fft2_r2c",
    "fft3",
    "fft3_c2r",
    "fft3_inplace",
    "fft3_r2c",
    "fft_c2r",
    "fft_inplace",
    "fft_r2c",
    "ifft",
    "ifft2",
    "ifft2_inplace",
    "ifft3",
    "ifft3_inplace",
    "ifft_inplace",
    "set_fft_plan_cache_size",
]

from .signal_processing.fast_fourier_transforms import (
    fft,
    fft2,
    fft2_c2r,
    fft2_inplace,
    fft2_r2c,
    fft3,
    fft3_c2r,
    fft3_inplace,
    fft3_r2c,
    fft_c2r,
    fft_inplace,
    fft_r2c,
    ifft,
    ifft2,
    ifft2_inplace,
    ifft3,
    ifft3_inplace,
    ifft_inplace,
    set_fft_plan_cache_size,
)

__all__ += ["fir", "iir"]

from .signal_processing.filter import fir, iir

__all__ += [
    "approx1",
    "approx1_uniform",
    "approx1_uniform_v2",
    "approx1_v2",
    "approx2",
    "approx2_uniform",
    "approx2_uniform_v2",
    "approx2_v2",
]

from .signal_processing.interpolation_and_approximation import (
    approx1,
    approx1_uniform,
    approx1_uniform_v2,
    approx1_v2,
    approx2,
    approx2_uniform,
    approx2_uniform_v2,
    approx2_v2,
)

# Unified API functions

__all__ += [
    "get_active_backend",
    "get_available_backends",
    "get_backend_count",
    "get_backend_id",
    "get_device_id",
    "set_backend",
]

from .unified_api_functions import (
    get_active_backend,
    get_available_backends,
    get_backend_count,
    get_backend_id,
    get_device_id,
    set_backend,
)

# Events

__all__ += ["AFEvent", "block_event", "create_event", "delete_event", "enqueue_wait_event", "mark_event"]

from .event_api import AFEvent, block_event, create_event, delete_event, enqueue_wait_event, mark_event

# Constants

__all__ += [
    "Match",
    "Moment",
    "Pad",
    "PointerSource",
    "TopK",
    "VarianceBias",
    "MatProp",
    "ImageFormat",
    "ConvMode",
    "ConvDomain",
    "ConvGradient",
]

from ._constants import (
    ConvDomain,
    ConvGradient,
    ConvMode,
    ImageFormat,
    Match,
    MatProp,
    Moment,
    Pad,
    PointerSource,
    TopK,
    VarianceBias,
)
