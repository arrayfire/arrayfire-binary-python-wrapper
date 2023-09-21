# flake8: noqa
from .version import ARRAYFIRE_VERSION, VERSION

__all__ = ["__version__"]
__version__ = VERSION

__all__ += ["__arrayfire_version__"]
__arrayfire_version__ = ARRAYFIRE_VERSION

__all__ += ["AFArray"]
from .defines import AFArray

__all__ = [
    "Dtype",
    "b8",
    "bool",
    "c32",
    "c64",
    "complex32",
    "complex64",
    "f16",
    "f32",
    "f64",
    "float16",
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "s16",
    "s32",
    "s64",
    "u8",
    "u16",
    "u32",
    "u64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]

from .dtypes import (
    Dtype,
    b8,
    bool,
    c32,
    c64,
    complex32,
    complex64,
    f16,
    f32,
    f64,
    float16,
    float32,
    float64,
    int16,
    int32,
    int64,
    s16,
    s32,
    s64,
    u8,
    u16,
    u32,
    u64,
    uint8,
    uint16,
    uint32,
    uint64,
)
