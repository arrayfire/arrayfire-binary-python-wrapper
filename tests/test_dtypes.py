import ctypes

import pytest

from arrayfire_wrapper.dtypes import Dtype
from arrayfire_wrapper.dtypes import bool as af_bool
from arrayfire_wrapper.dtypes import (
    complex128,
    float32,
    float64,
    implicit_dtype,
    int8,
    int32,
    int64,
    str_to_dtype,
    uint16,
)


def test_dtype_str_representation() -> None:
    assert str(float32) == "float32"


def test_dtype_repr_representation() -> None:
    assert repr(float32) == "arrayfire.float32(typecode<f>)"


def test_dtype_equality() -> None:
    dt1 = Dtype("float32", "f", ctypes.c_float, "float", 0)
    dt2 = float32
    assert dt1 == dt2


def test_dtype_inequality() -> None:
    assert float32 != int8


@pytest.mark.parametrize(
    "number,array_dtype,expected_dtype",
    [
        (1, int64, int64),
        (1.0, float64, float64),
        (1.0, float32, float32),
        (True, float32, af_bool),
        (1 + 2j, complex128, complex128),
    ],
)
def test_implicit_dtype(number: int | float | bool | complex, array_dtype: Dtype, expected_dtype: Dtype) -> None:
    result_dtype = implicit_dtype(number, array_dtype)
    assert result_dtype == expected_dtype


def test_implicit_dtype_raises_error_invalid_array_dtype() -> None:
    with pytest.raises(TypeError):
        implicit_dtype([1], "invalid_dtype")  # type: ignore[arg-type]


def test_implicit_dtype_raises_error_invalid_number_type() -> None:
    with pytest.raises(TypeError):
        implicit_dtype("invalid_number", int32)  # type: ignore[arg-type]


def test_implicit_dtype_raises_error_invalid_combination() -> None:
    with pytest.raises(TypeError):
        implicit_dtype("invalid_number", float32)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value,expected_dtype",
    [
        ("i8", int8),
        ("int", int32),
        ("uint16", uint16),
        ("float", float32),
    ],
)
def test_str_to_dtype(value: str, expected_dtype: Dtype) -> None:
    result_dtype = str_to_dtype(value)
    assert result_dtype == expected_dtype


def test_str_to_dtype_raises_error() -> None:
    with pytest.raises(TypeError):
        str_to_dtype("invalid_dtype")


def test_str_to_dtype_raises_error_case_insensitive() -> None:
    with pytest.raises(TypeError):
        str_to_dtype("Int")
