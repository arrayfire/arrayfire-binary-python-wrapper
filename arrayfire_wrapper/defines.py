from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass
from typing import Type, TypeVar

# def is_arch_x86() -> bool:
#     machine = platform.machine()
#     return platform.architecture()[0][0:2] == "32" and (machine[-2:] == "86" or machine[0:3] == "arm")


# BUG
def is_arch_x86() -> bool:
    """
    Check if the current system architecture is 32-bit and either x86 or ARM.

    Returns:
        bool: True if the architecture is 32-bit and x86 or ARM, False otherwise.
    """
    architecture_bits, _ = platform.architecture()
    machine = platform.machine()

    is_32_bit = architecture_bits.startswith("32")
    is_x86_or_arm = machine.endswith("86") or machine.startswith("arm")

    return is_32_bit and is_x86_or_arm


# Define a generic type variable for the base class
T = TypeVar("T", bound=ctypes.c_void_p)


class _AFBase(ctypes.c_void_p):
    @classmethod
    def create_null_pointer(cls: Type[T]) -> T:
        cls.value = None
        return cls()


class AFArray(_AFBase):
    pass


CType = Type[ctypes._SimpleCData]
CDimT = ctypes.c_int if is_arch_x86() else ctypes.c_longlong


@dataclass(frozen=True)
class ArrayBuffer:
    address: int
    length: int = 0


class CShape(tuple):
    def __new__(cls, *args: int) -> CShape:
        cls.original_shape = len(args)
        return tuple.__new__(cls, args)

    def __init__(self, x1: int = 1, x2: int = 1, x3: int = 1, x4: int = 1) -> None:
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.x1, self.x2, self.x3, self.x4}"

    @property
    def c_array(self) -> ctypes.Array:
        c_shape = CDimT * 4  # ctypes.c_int | ctypes.c_longlong * 4
        return c_shape(CDimT(self.x1), CDimT(self.x2), CDimT(self.x3), CDimT(self.x4))
