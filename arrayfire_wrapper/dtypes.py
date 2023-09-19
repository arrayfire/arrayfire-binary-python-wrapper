from __future__ import annotations

import ctypes
from dataclasses import dataclass

from .defines import CType

_python_bool = bool


@dataclass(frozen=True)
class Dtype:
    name: str
    typecode: str
    c_type: CType
    typename: str
    c_api_value: int  # Internal use only

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"arrayfire.{self.name}(typecode<{self.typecode}>)"


# Specification required
int8 = Dtype("int8", "i8", ctypes.c_char, "int8", 4)  # HACK int8 - Not Supported, b8?
int16 = Dtype("int16", "h", ctypes.c_short, "short int", 10)
int32 = Dtype("int32", "i", ctypes.c_int, "int", 5)
int64 = Dtype("int64", "l", ctypes.c_longlong, "long int", 8)
uint8 = Dtype("uint8", "B", ctypes.c_ubyte, "unsigned_char", 7)
uint16 = Dtype("uint16", "H", ctypes.c_ushort, "unsigned short int", 11)
uint32 = Dtype("uint32", "I", ctypes.c_uint, "unsigned int", 6)
uint64 = Dtype("uint64", "L", ctypes.c_ulonglong, "unsigned long int", 9)
float16 = Dtype("float16", "e", ctypes.c_uint16, "half", 12)
float32 = Dtype("float32", "f", ctypes.c_float, "float", 0)
float64 = Dtype("float64", "d", ctypes.c_double, "double", 2)
complex64 = Dtype("complex64", "F", ctypes.c_float * 2, "float complex", 1)  # type: ignore[arg-type]
complex128 = Dtype("complex128", "D", ctypes.c_double * 2, "double complex", 3)  # type: ignore[arg-type]
bool = Dtype("bool", "b", ctypes.c_bool, "bool", 4)

supported_dtypes = (
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    complex64,
    complex128,
    bool,
    int8,  # BUG if place on top of the list
)


def is_complex_dtype(dtype: Dtype) -> _python_bool:
    return dtype in {complex64, complex128}


def to_str(c_str: ctypes.c_char_p | ctypes.Array[ctypes.c_char]) -> str:
    return str(c_str.value.decode("utf-8"))  # type: ignore[union-attr]


def implicit_dtype(number: int | float | _python_bool | complex, array_dtype: Dtype) -> Dtype:
    if isinstance(number, _python_bool):
        number_dtype = bool
    elif isinstance(number, int):
        number_dtype = int64
    elif isinstance(number, float):
        number_dtype = float64
    elif isinstance(number, complex):
        number_dtype = complex128
    else:
        raise TypeError(f"{type(number)} is not supported and can not be converted to af.Dtype.")

    if not (array_dtype == float32 or array_dtype == complex64):
        return number_dtype

    if number_dtype == float64:
        return float32

    if number_dtype == complex128:
        return complex64

    return number_dtype


def c_api_value_to_dtype(value: int) -> Dtype:
    for dtype in supported_dtypes:
        if value == dtype.c_api_value:
            return dtype

    raise TypeError("There is no supported dtype that matches passed dtype C API value.")


def str_to_dtype(value: str) -> Dtype:
    for dtype in supported_dtypes:
        if value == dtype.typecode or value == dtype.typename or value == dtype.name:
            return dtype

    raise TypeError("There is no supported dtype that matches passed dtype typecode.")
