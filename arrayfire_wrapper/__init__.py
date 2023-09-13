# flake8: noqa
from .version import ARRAYFIRE_VERSION, VERSION

__all__ = ["__version__"]
__version__ = VERSION

__all__ += ["__arrayfire_version__"]
__arrayfire_version__ = ARRAYFIRE_VERSION

__all__ += ["add"]
from arrayfire_wrapper.library.mathematical_functions import add

__all__ += ["randu"]
from arrayfire_wrapper.library.create_and_modify_array import randu
