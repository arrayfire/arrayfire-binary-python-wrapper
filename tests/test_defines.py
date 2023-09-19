import ctypes

from arrayfire_wrapper.defines import AFArray


def test_null_pointer_value() -> None:
    assert AFArray.create_null_pointer().value == AFArray(0).value == ctypes.c_void_p(0).value
