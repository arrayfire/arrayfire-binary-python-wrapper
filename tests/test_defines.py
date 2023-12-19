import ctypes
from unittest.mock import patch

import pytest

from arrayfire_wrapper.defines import AFArray, ArrayBuffer, CDimT, CShape, _AFBase, is_arch_x86


def test_null_pointer_value() -> None:
    assert AFArray.create_null_pointer().value == AFArray(0).value == ctypes.c_void_p(0).value


def test_af_base_create_null_pointer() -> None:
    af_base = _AFBase()
    null_pointer = af_base.create_null_pointer()
    assert isinstance(null_pointer, _AFBase)


def test_af_array_create_null_pointer() -> None:
    af_array = AFArray()
    null_pointer = af_array.create_null_pointer()
    assert isinstance(null_pointer, AFArray)


def test_array_buffer_creation() -> None:
    array_buffer = ArrayBuffer(address=0x1000, length=10)

    assert array_buffer.address == 0x1000
    assert array_buffer.length == 10


def test_array_buffer_immutable() -> None:
    array_buffer = ArrayBuffer(address=0x2000, length=5)

    with pytest.raises(AttributeError):
        array_buffer.address = 0x3000  # type: ignore[misc]

    with pytest.raises(AttributeError):
        array_buffer.length = 8  # type: ignore[misc]


def test_cshape_creation() -> None:
    c_shape = CShape(1, 2, 3, 4)
    assert c_shape.x1 == 1
    assert c_shape.x2 == 2
    assert c_shape.x3 == 3
    assert c_shape.x4 == 4


def test_cshape_repr() -> None:
    c_shape = CShape(1, 2, 3, 4)
    assert repr(c_shape) == "CShape(1, 2, 3, 4)"


def test_cshape_c_array() -> None:
    c_shape = CShape(1, 2, 3, 4)
    c_array = c_shape.c_array
    assert isinstance(c_array, ctypes.Array)
    assert len(c_array) == 4
    assert c_array[0] == CDimT(1).value
    assert c_array[1] == CDimT(2).value
    assert c_array[2] == CDimT(3).value
    assert c_array[3] == CDimT(4).value


@pytest.mark.parametrize(
    "architecture, machine, expected_result",
    [
        ("32bit", "x86", True),
        ("32bit", "arm", True),
        ("64bit", "x86", False),
        ("32bit", "other", False),
        ("64bit", "other", False),
    ],
)
def test_is_arch_x86(architecture: str, machine: str, expected_result: bool) -> None:
    with patch("platform.architecture", lambda: (architecture, "")), patch("platform.machine", lambda: machine):
        assert is_arch_x86() == expected_result
