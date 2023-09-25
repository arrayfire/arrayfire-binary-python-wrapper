from __future__ import annotations

import ctypes

# TODO add more features


class AFFeatures(ctypes.c_void_p):
    """
    source: https://arrayfire.org/docs/features_8h.htm#a294c8f0e20b10dfc4f4f18566dba06bc
    """

    @classmethod
    def create_null_pointer(cls) -> AFFeatures:
        cls.value = None
        return cls()
