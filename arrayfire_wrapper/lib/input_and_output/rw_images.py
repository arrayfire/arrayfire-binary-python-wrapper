import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._constants import ImageFormat
from arrayfire_wrapper.lib._utility import call_from_clib


def delete_image_memory(image: AFArray, /) -> None:
    """
    source: https://arrayfire.org/docs/group__imagemem__func__delete.htm#gab4a7f3417baf7287371d295b292b93bb
    """
    call_from_clib(delete_image_memory.__name__, ctypes.pointer(image))


def is_image_io_available() -> bool:
    """
    source: https://arrayfire.org/docs/group__imageio__func__available.htm#ga2d7fa02d9009a3ca6f16bfb1a5aacd0b
    """
    out = ctypes.c_bool(False)
    call_from_clib(is_image_io_available.__name__, ctypes.pointer(out))
    return bool(out.value)


def load_image(filename: str, is_color: bool, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__imageio__func__load.htm#ga9c505bba21cd2d5aa277ad1e6f0ffb5f
    """
    out = AFArray.create_null_pointer()
    call_from_clib(load_image.__name__, ctypes.pointer(out), filename.encode("utf-8"), is_color)
    return out


def load_image_native(filename: str, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__imageio__func__load.htm#gaf8ccff53540bcc78ab814864f3c74ded
    """
    out = AFArray.create_null_pointer()
    call_from_clib(load_image_native.__name__, ctypes.pointer(out), filename.encode("utf-8"))
    return out


def load_image_memory(pointer: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__imagemem__func__load.htm#gadcf65c240a2956afb296194685c4f7c5
    """
    out = AFArray.create_null_pointer()
    call_from_clib(load_image_memory.__name__, ctypes.pointer(out), ctypes.c_void_p(pointer))
    return out


def save_image(image: AFArray, filename: str, /) -> None:
    """
    source: https://arrayfire.org/docs/group__imageio__func__save.htm#ga2d73aad096dd1e0022fe7369112168b8
    """
    call_from_clib(save_image.__name__, ctypes.c_char_p(filename.encode("ascii")), image)
    return None


def save_image_native(image: AFArray, filename: str, /) -> None:
    """
    source: https://arrayfire.org/docs/group__imageio__func__save.htm#gac19d3cc88b12d0ea8b9bc751927f5c83
    """
    call_from_clib(save_image_native.__name__, ctypes.c_char_p(filename.encode("ascii")), image)
    return None


def save_image_memory(pointer: int, image: AFArray, image_format: ImageFormat) -> None:
    """
    source: https://arrayfire.org/docs/group__imagemem__func__save.htm#gae9f582ee747e6ac5c75209dc6224be8a
    """
    call_from_clib(save_image_memory.__name__, ctypes.pointer(ctypes.c_void_p(pointer)), image, image_format.value)
