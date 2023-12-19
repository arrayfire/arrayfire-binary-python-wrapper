import ctypes

import pytest

from arrayfire_wrapper import complex32, complex64, float64, int16, int64, uint8
from arrayfire_wrapper.dtypes import Dtype
from arrayfire_wrapper.dtypes import bool as afbool
from arrayfire_wrapper.dtypes import (
    c_api_value_to_dtype,
    float32,
    implicit_dtype,
    int32,
    s16,
    str_to_dtype,
    to_str,
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
    assert float32 != int32


@pytest.mark.parametrize(
    "value,expected_dtype",
    [
        ("short int", s16),
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


def test_to_str_with_c_char_p() -> None:
    c_str = ctypes.c_char_p(b"Hello, World!")
    result = to_str(c_str)
    assert result == "Hello, World!"


def test_to_str_with_c_array() -> None:
    c_str_array = (ctypes.c_char * 13)(*b"Hello, World!")  # type: ignore[misc]
    result = to_str(c_str_array)
    assert result == "Hello, World!"


def test_to_str_with_invalid_input() -> None:
    with pytest.raises(AttributeError):
        invalid_input = "not_a_c_type"
        to_str(invalid_input)  # type: ignore[arg-type]


def test_implicit_dtype_bool() -> None:
    result: Dtype = implicit_dtype(True, int32)
    assert result == afbool


def test_implicit_dtype_int() -> None:
    result: Dtype = implicit_dtype(42, float32)
    assert result == int64


def test_implicit_dtype_float() -> None:
    result: Dtype = implicit_dtype(3.14, complex32)
    assert result == float32


def test_implicit_dtype_complex() -> None:
    result: Dtype = implicit_dtype(1 + 2j, complex32)
    assert result == complex64


def test_implicit_dtype_invalid_type() -> None:
    with pytest.raises(TypeError):
        implicit_dtype("not_supported", float32)  # type: ignore[arg-type]


def test_implicit_dtype_non_float_array_dtype() -> None:
    result: Dtype = implicit_dtype(42, int64)
    assert result == int64


def test_implicit_dtype_float_array_dtype() -> None:
    result: Dtype = implicit_dtype(3.14, float64)
    assert result == float32


def test_implicit_dtype_complex_array_dtype() -> None:
    result: Dtype = implicit_dtype(1 + 2j, complex64)
    assert result == complex64


def test_implicit_dtype_none_input() -> None:
    with pytest.raises(TypeError):
        implicit_dtype(None, float32)  # type: ignore[arg-type]


def test_implicit_dtype_none_array_dtype() -> None:
    with pytest.raises(ValueError):
        implicit_dtype(42, None)  # type: ignore[arg-type]


def test_implicit_dtype_mixed_complex_array_dtype() -> None:
    result: Dtype = implicit_dtype(1 + 2j, float32)
    assert result == complex64


def test_implicit_dtype_invalid_array_dtype() -> None:
    with pytest.raises(ValueError):
        implicit_dtype(42, "invalid_dtype")  # type: ignore[arg-type]


def test_implicit_dtype_mixed_array_dtype() -> None:
    result: Dtype = implicit_dtype(3.14, complex32)
    assert result == float32


def test_implicit_dtype_bool_array_dtype() -> None:
    result: Dtype = implicit_dtype(True, afbool)
    assert result == afbool


def test_implicit_dtype_large_integer() -> None:
    result: Dtype = implicit_dtype(2**64, int32)
    assert result == int64


def test_c_api_value_to_dtype_valid() -> None:
    for dtype in [int16, int32, int64, uint8]:
        result: Dtype = c_api_value_to_dtype(dtype.c_api_value)
        assert result == dtype


def test_c_api_value_to_dtype_invalid() -> None:
    with pytest.raises(TypeError, match="There is no supported dtype"):
        c_api_value_to_dtype(-1)
