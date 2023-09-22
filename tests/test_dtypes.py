import ctypes

import pytest

from arrayfire_wrapper.dtypes import Dtype, float32, int32, s16, str_to_dtype, uint16


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
