__all__ = ["BackendType"]

import ctypes
import enum
import os
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from ._logger import logger
from .defines import is_arch_x86
from .version import ARRAYFIRE_VER_MAJOR


class _LibPrefixes(Enum):
    forge = ""
    arrayfire = "af"


class _SupportedPlatforms(Enum):
    windows = "Windows"
    darwin = "Darwin"  # OSX
    linux = "Linux"

    @classmethod
    def is_cygwin(cls, name: str) -> bool:
        return "cyg" in name.lower()


@dataclass(frozen=True)
class _BackendPathConfig:
    lib_prefix: str
    lib_postfix: str
    af_path: Path
    cuda_found: bool

    def __iter__(self) -> Iterator:
        return iter((self.lib_prefix, self.lib_postfix, self.af_path, self.af_path, self.cuda_found))


def _get_backend_path_config() -> _BackendPathConfig:
    platform_name = platform.system()
    cuda_found = False

    try:
        af_path = Path(os.environ["AF_PATH"])
    except KeyError:
        af_path = None

    try:
        cuda_path = Path(os.environ["CUDA_PATH"])
    except KeyError:
        cuda_path = None

    if platform_name == _SupportedPlatforms.windows.value or _SupportedPlatforms.is_cygwin(platform_name):
        if platform_name == _SupportedPlatforms.windows.value:
            # HACK Supressing crashes caused by missing dlls
            # http://stackoverflow.com/questions/8347266/missing-dll-print-message-instead-of-launching-a-popup
            # https://msdn.microsoft.com/en-us/_clib/windows/desktop/ms680621.aspx
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)  # type: ignore[attr-defined]

        if not af_path:
            af_path = _find_default_path(f"C:/Program Files/ArrayFire/v{ARRAYFIRE_VER_MAJOR}")

        if cuda_path and (cuda_path / "bin").is_dir() and (cuda_path / "nvvm/bin").is_dir():
            cuda_found = True

        return _BackendPathConfig("", ".dll", af_path, cuda_found)

    if platform_name == _SupportedPlatforms.darwin.value:
        default_cuda_path = Path("/usr/local/cuda/")

        if not af_path:
            af_path = _find_default_path("/opt/arrayfire", "/usr/local")

        if not (cuda_path and default_cuda_path.exists()):
            cuda_found = (default_cuda_path / "lib").is_dir() and (default_cuda_path / "/nvvm/lib").is_dir()

        return _BackendPathConfig("lib", f".{ARRAYFIRE_VER_MAJOR}.dylib", af_path, cuda_found)

    if platform_name == _SupportedPlatforms.linux.value:
        default_cuda_path = Path("/usr/local/cuda/")

        if not af_path:
            af_path = _find_default_path(f"/opt/arrayfire-{ARRAYFIRE_VER_MAJOR}", "/opt/arrayfire/", "/usr/local/")

        if not (cuda_path and default_cuda_path.exists()):
            if "64" in platform.architecture()[0]:  # Check either is 64 bit arch is selected
                cuda_found = (default_cuda_path / "lib64").is_dir() and (default_cuda_path / "nvvm/lib64").is_dir()
            else:
                cuda_found = (default_cuda_path / "lib").is_dir() and (default_cuda_path / "nvvm/lib").is_dir()

        return _BackendPathConfig("lib", f".so.{ARRAYFIRE_VER_MAJOR}", af_path, cuda_found)

    raise OSError(f"{platform_name} is not supported.")


def _find_default_path(*args: str) -> Path:
    for path in args:
        default_path = Path(path)
        if default_path.exists():
            return default_path
    raise ValueError("None of specified default paths were found.")


class BackendType(enum.Enum):  # TODO change name - avoid using _backend_type - e.g. type
    unified = 0  # NOTE It is set as Default value on Arrayfire backend
    cpu = 1
    cuda = 2
    opencl = 4
    oneapi = 8

    def __iter__(self) -> Iterator:
        # NOTE cpu comes last because we want to keep this order priorty during backend initialization
        return iter((self.unified, self.cuda, self.opencl, self.cpu))


class Backend:
    _backend_type: BackendType
    # HACK for osx
    # _backend.clib = ctypes.CDLL("/opt/arrayfire//lib/libafcpu.3.dylib")
    # HACK for windows
    # _backend.clib = ctypes.CDLL("C:/Program Files/ArrayFire/v3/lib/afcpu.dll")
    _clib: ctypes.CDLL

    def __init__(self) -> None:
        self._backend_path_config = _get_backend_path_config()

        self._load_forge_lib()
        self._load_backend_libs()

    def _load_forge_lib(self) -> None:
        for lib_name in self._lib_names("forge", _LibPrefixes.forge):
            try:
                ctypes.cdll.LoadLibrary(str(lib_name))
                logger.info(f"Loaded {lib_name}")
                break
            except OSError:
                logger.warning(f"Unable to load {lib_name}")
                pass

    def _load_backend_libs(self) -> None:
        for backend_type in BackendType:
            self._load_backend_lib(backend_type)

            if self._backend_type:
                logger.info(f"Setting {backend_type.name} as backend.")
                break

        if not self._backend_type and not self._clib:
            raise RuntimeError(
                "Could not load any ArrayFire libraries.\n"
                "Please look at https://github.com/arrayfire/arrayfire-python/wiki for more information."
            )

    def _load_backend_lib(self, _backend_type: BackendType) -> None:
        # NOTE we still set unified cdll to it's original name later, even if the path search is different
        name = _backend_type.name if _backend_type != BackendType.unified else ""

        for lib_name in self._lib_names(name, _LibPrefixes.arrayfire):
            try:
                ctypes.cdll.LoadLibrary(str(lib_name))
                self._backend_type = _backend_type
                self._clib = ctypes.CDLL(str(lib_name))

                if _backend_type == BackendType.cuda:
                    self._load_nvrtc_builtins_lib(lib_name.parent)

                logger.info(f"Loaded {lib_name}")
                break
            except OSError:
                logger.warning(f"Unable to load {lib_name}")
                pass

    def _load_nvrtc_builtins_lib(self, lib_path: Path) -> None:
        nvrtc_name = self._find_nvrtc_builtins_lib_name(lib_path)
        if nvrtc_name:
            ctypes.cdll.LoadLibrary(str(lib_path / nvrtc_name))
            logger.info(f"Loaded {lib_path / nvrtc_name}")
        else:
            logger.warning("Could not find local nvrtc-builtins library")

    def _lib_names(self, name: str, lib: _LibPrefixes, ver_major: str | None = None) -> list[Path]:
        post = self._backend_path_config.lib_postfix if ver_major is None else ver_major
        lib_name = self._backend_path_config.lib_prefix + lib.value + name + post

        lib64_path = self._backend_path_config.af_path / "lib64"
        search_path = lib64_path if lib64_path.is_dir() else self._backend_path_config.af_path / "lib"

        site_path = Path(sys.prefix) / "lib64" if not is_arch_x86() else Path(sys.prefix) / "lib"

        # prefer locally packaged arrayfire libraries if they exist
        af_module = __import__(__name__)
        local_path = Path(af_module.__path__[0]) if af_module.__path__ else Path("")

        lib_paths = [Path("", lib_name), site_path / lib_name, local_path / lib_name]

        if self._backend_path_config.af_path:  # prefer specified AF_PATH if exists
            return [search_path / lib_name] + lib_paths
        else:
            lib_paths.insert(2, Path(str(search_path), lib_name))
            return lib_paths

    def _find_nvrtc_builtins_lib_name(self, search_path: Path) -> str | None:
        for f in search_path.iterdir():
            if "nvrtc-builtins" in f.name:
                return f.name
        return None

    @property
    def backend_type(self) -> BackendType:
        return self._backend_type

    @property
    def clib(self) -> ctypes.CDLL:
        return self._clib


# Initialize the backend
_backend = Backend()


def get_backend() -> Backend:
    """
    Get the current active backend.

    Returns
    -------
    value : Backend
        Current active backend.
    """

    return _backend
