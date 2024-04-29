import arrayfire_wrapper.lib.interface_functions.opencl as cl


def test_get_context_type():
    assert isinstance(cl.get_context(), int)


def test_get_queue_type():
    assert isinstance(cl.get_queue(), int)


def test_get_device_id():
    assert isinstance(cl.get_device_id(), int)


def test_set_device_id():
    cl.set_device_id(0)
    assert cl.get_device_id() == 0


def test_get_device_type():
    assert cl.get_device_type() == cl.DeviceType.GPU  # change according to device


def test_get_platform():
    assert cl.get_platform() == cl.PlatformType.INTEL  # change according to platform
