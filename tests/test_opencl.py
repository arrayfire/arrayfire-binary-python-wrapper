import ctypes

import arrayfire_wrapper.lib.interface_functions.opencl as cl


def test_get_context_type() -> None:
    ptr = cl.get_context()
    assert isinstance(ptr, ctypes.c_void_p)


def test_get_queue_type() -> None:
    assert isinstance(cl.get_queue(), ctypes.c_void_p)


def test_get_device_id() -> None:
    assert isinstance(cl.get_device_id(), int)


def test_set_device_id() -> None:
    cl.set_device_id(0)
    assert cl.get_device_id() == 0


def test_get_device_type() -> None:
    assert cl.get_device_type() == cl.DeviceType.GPU  # change according to device


def test_get_platform() -> None:
    assert cl.get_platform() == cl.PlatformType.INTEL  # change according to platform
