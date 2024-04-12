__all__ = ["BackendType", "get_backend", "set_backend"]

import ctypes
import enum
import os
import platform
import sys
import sysconfig
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from .version import ARRAYFIRE_VER_MAJOR

VERBOSE_LOADS = os.environ.get("AF_VERBOSE_LOADS", "") == "1"


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
    af_path: Path | None
    af_is_user_path: bool
    cuda_found: bool

    def __iter__(self) -> Iterator:
        return iter((self.lib_prefix, self.lib_postfix, self.af_path, self.af_path, self.cuda_found))


def _get_backend_path_config() -> _BackendPathConfig:
    platform_name = platform.system()
    cuda_found = False

    # try to use user provided AF_PATH if explicitly set
    try:
        af_path = Path(os.environ["AF_PATH"])
        af_is_user_path = True
    except KeyError:
        af_path = None
        af_is_user_path = False

    try:
        cuda_path = Path(os.environ["CUDA_PATH"])
    except KeyError:
        cuda_path = None

    # Try to find default arrayfire installation paths
    if platform_name == _SupportedPlatforms.windows.value or _SupportedPlatforms.is_cygwin(platform_name):
        if platform_name == _SupportedPlatforms.windows.value:
            # HACK Supressing crashes caused by missing dlls
            # http://stackoverflow.com/questions/8347266/missing-dll-print-message-instead-of-launching-a-popup
            # https://msdn.microsoft.com/en-us/_clib/windows/desktop/ms680621.aspx
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)  # type: ignore[attr-defined]

        if not af_path:
            try:
                af_path = _find_default_path(f"C:/Program Files/ArrayFire/v{ARRAYFIRE_VER_MAJOR}")
            except ValueError:
                af_path = None

        if cuda_path and (cuda_path / "bin").is_dir() and (cuda_path / "nvvm/bin").is_dir():
            cuda_found = True

        return _BackendPathConfig("", ".dll", af_path, af_is_user_path, cuda_found)

    if platform_name == _SupportedPlatforms.darwin.value:
        default_cuda_path = Path("/usr/local/cuda/")

        if not af_path:
            af_path = _find_default_path("/opt/arrayfire", "/usr/local")
            try:
                af_path = _find_default_path(
                    f"C:/Program Files/ArrayFire/v{ARRAYFIRE_VER_MAJOR}",
                    "C:/Program Files (x86)/ArrayFire/v{ARRAYFIRE_VER_MAJOR}",
                )
            except ValueError:
                af_path = None

        if not (cuda_path and default_cuda_path.exists()):
            cuda_found = (default_cuda_path / "lib").is_dir() and (default_cuda_path / "/nvvm/lib").is_dir()

        return _BackendPathConfig("lib", f".{ARRAYFIRE_VER_MAJOR}.dylib", af_path, af_is_user_path, cuda_found)

    if platform_name == _SupportedPlatforms.linux.value:
        default_cuda_path = Path("/usr/local/cuda/")

        if not af_path:
            try:
                af_path = _find_default_path(f"/opt/arrayfire-{ARRAYFIRE_VER_MAJOR}", "/opt/arrayfire/", "/usr/local/")
            except ValueError:
                af_path = None

        if not (cuda_path and default_cuda_path.exists()):
            if "64" in platform.architecture()[0]:  # Check either is 64 bit arch is selected
                cuda_found = (default_cuda_path / "lib64").is_dir() and (default_cuda_path / "nvvm/lib64").is_dir()
            else:
                cuda_found = (default_cuda_path / "lib").is_dir() and (default_cuda_path / "nvvm/lib").is_dir()

        return _BackendPathConfig("lib", f".so.{ARRAYFIRE_VER_MAJOR}", af_path, af_is_user_path, cuda_found)

    raise OSError(f"{platform_name} is not supported.")


# finds paths to locally packaged arrayfire libraries if they exist in site
def _find_site_local_path() -> Path:
    local_paths = ["."]

    # module search paths
    af_module = __import__(__name__)
    module_paths = af_module.__path__ if af_module.__path__ else []
    for path in module_paths:
        local_paths.append(path)

    # site search path
    purelib_path = sysconfig.get_path("purelib")
    platlib_path = sysconfig.get_path("platlib")
    local_paths.append(purelib_path)
    local_paths.append(platlib_path)

    # sys search path
    local_paths.extend(sys.path)

    module_name = af_module.__name__
    for path in local_paths:
        lpath = Path(path)
        if lpath.exists():
            p = lpath.glob(f"{module_name}/binaries/*")
            files = [x.name for x in p if x.is_file()]
            query_libnames = ["afcpu", "afoneapi", "afopencl", "afcuda", "af", "forge"]
            found_lib_in_dir = any(q in f for q in query_libnames for f in files)
            if found_lib_in_dir:
                if VERBOSE_LOADS:
                    print(lpath)
                    print(lpath / module_name / "binaries")
                return lpath / module_name / "binaries"
    raise RuntimeError("No binaries detected in site path.")


def _find_default_path(*args: str) -> Path:
    for path in args:
        default_path = Path(path)
        if default_path.exists():
            return default_path
    raise ValueError("None of specified default paths were found.")


class BackendType(enum.Enum):  # TODO change name - avoid using _backend_type - e.g. type
    cuda = 2
    opencl = 4
    oneapi = 8
    cpu = 1
    unified = 0  # NOTE It is set as Default value on Arrayfire backend

    def __iter__(self) -> Iterator:
        # NOTE cpu comes last because we want to keep this order priorty during backend initialization
        return iter((self.cuda, self.opencl, self.oneapi, self.cpu, self.unified))


class Backend:
    _backend_type: BackendType | None
    _clibs: dict[BackendType, ctypes.CDLL]

    def __init__(self) -> None:
        self._backend_path_config = _get_backend_path_config()

        self._backend_type = None
        self._clibs = {}
        self._load_backend_libs()
        self._load_forge_lib()

    def _change_backend(self, backend_type: BackendType) -> None:
        # if unified is available, do dynamic module loading through libaf
        if self._backend_type == BackendType.unified:
            from arrayfire_wrapper.lib.unified_api_functions import set_backend as unified_set_backend

            try:
                unified_set_backend(backend_type)
            except RuntimeError as e:
                print(f"Unable to change backend using unified loader: {str(e)}")
        # if unified not available
        else:
            if backend_type in self._clibs:
                self._backend_type = backend_type
            else:
                self._backend_path_config = _get_backend_path_config()
                self._load_backend_libs(backend_type)
                # self._load_forge_lib() # needed to reload?

    def _load_forge_lib(self) -> None:
        for lib_name in self._lib_names("forge", _LibPrefixes.forge):
            try:
                ctypes.cdll.LoadLibrary(str(lib_name))
                if VERBOSE_LOADS:
                    print(f"Loaded {lib_name}")
                break
            except OSError:
                if VERBOSE_LOADS:
                    print(f"Unable to load {lib_name}")
                pass

    def _load_backend_libs(self, specific_backend: BackendType | None = None) -> None:
        available_backends = [specific_backend] if specific_backend else list(BackendType)
        for backend_type in available_backends:
            self._load_backend_lib(backend_type)

            if self._backend_type:
                if VERBOSE_LOADS:
                    print(f"Setting {backend_type.name} as backend.")
                break

        if not self._backend_type and not self._clibs:
            raise RuntimeError(
                "Could not load any ArrayFire libraries.\n"
                "Please look at https://github.com/arrayfire/arrayfire-python/wiki for more information."
            )

    def _load_backend_lib(self, _backend_type: BackendType) -> None:
        # NOTE we still set unified cdll to it's original name later, even if the path search is different
        name = _backend_type.name if _backend_type != BackendType.unified else ""

        for lib_name in self._lib_names(name, _LibPrefixes.arrayfire):
            try:
                if VERBOSE_LOADS:
                    print(f"Attempting to load {lib_name}")
                ctypes.cdll.LoadLibrary(str(lib_name))
                self._backend_type = _backend_type
                self._clibs[_backend_type] = ctypes.CDLL(str(lib_name))

                if _backend_type == BackendType.cuda:
                    self._load_nvrtc_builtins_lib(lib_name.parent)

                if VERBOSE_LOADS:
                    print(f"Loaded {lib_name}")
                break
            except OSError:
                if VERBOSE_LOADS:
                    print(f"Unable to load {lib_name}")
                pass

    def _load_nvrtc_builtins_lib(self, lib_path: Path) -> None:
        nvrtc_name = self._find_nvrtc_builtins_lib_name(lib_path)
        if nvrtc_name:
            ctypes.cdll.LoadLibrary(str(lib_path / nvrtc_name))
            if VERBOSE_LOADS:
                print(f"Loaded {lib_path / nvrtc_name}")
        else:
            if VERBOSE_LOADS:
                print("Could not find local nvrtc-builtins library")

    def _lib_names(self, name: str, lib: _LibPrefixes, ver_major: str | None = None) -> list[Path]:
        post = self._backend_path_config.lib_postfix if ver_major is None else ver_major
        lib_name = self._backend_path_config.lib_prefix + lib.value + name + post

        lib_paths = [Path(lib_name)]

        # use local or site packaged arrayfire libraries if they exist
        try:
            local_path = _find_site_local_path()
            lib_paths.append(local_path / lib_name)
        except RuntimeError as e:
            if VERBOSE_LOADS:
                print(f"Moving on to system libraries, site local load failed due to: {str(e)}")
            pass

        if self._backend_path_config.af_path:  # prefer specified AF_PATH if exists
            lib64_path = self._backend_path_config.af_path / "lib64"
            search_path = lib64_path if lib64_path.is_dir() else self._backend_path_config.af_path / "lib"
            # prefer path explicitly set by user through AF_PATH
            if self._backend_path_config.af_is_user_path:
                return [search_path / lib_name] + lib_paths
            # otherwise, prefer to use site-packaged or local path
            return lib_paths + [search_path / lib_name]

        return lib_paths

    def _find_nvrtc_builtins_lib_name(self, search_path: Path) -> str | None:
        for f in search_path.iterdir():
            if "nvrtc-builtins" in f.name:
                return f.name
        return None

    @property
    def backend_type(self) -> BackendType:
        if self._backend_type:
            return self._backend_type
        raise RuntimeError("No valid _backend_type")

    @property
    def clib(self) -> ctypes.CDLL:
        if self._backend_type:
            return self._clibs[self._backend_type]
        raise RuntimeError("No valid _backend_type")


# Initialize the backend
__backend = Backend()


def get_backend() -> Backend:
    """
    Get the current active backend.

    Returns
    -------
    value : Backend
        Current active backend.
    """

    return __backend


def set_backend(backend_type: BackendType) -> None:
    try:
        backend = get_backend()
        backend._change_backend(backend_type)
    except RuntimeError:
        print(f"Requested backend {backend_type.name} could not be found")
