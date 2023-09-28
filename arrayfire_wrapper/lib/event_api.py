from __future__ import annotations

import ctypes

from arrayfire_wrapper.defines import _AFBase
from arrayfire_wrapper.lib._utility import call_from_clib


class AFEvent(_AFBase):
    pass


def block_event(event: AFEvent, /) -> None:
    """
    source: https://arrayfire.org/docs/group__event__api.htm#ga4149dff7f1cfd70347d1418cd2b9eb5c
    """
    call_from_clib(block_event.__name__, event.value)


def create_event(event: AFEvent, /) -> None:
    """
    source: https://arrayfire.org/docs/group__event__api.htm#gaea827514b09b8d92f9dd218dd8310b7a
    """
    call_from_clib(create_event.__name__, ctypes.pointer(event))


def delete_event(event: AFEvent, /) -> None:
    """
    source: https://arrayfire.org/docs/group__event__api.htm#ga8fd20fbc146d53606f83a722dd5221c3
    """
    call_from_clib(delete_event.__name__, event)


def enqueue_wait_event(event: AFEvent, /) -> None:
    """
    source: https://arrayfire.org/docs/group__event__api.htm#gac26df9220143fcdd07647bf65bf31a34
    """
    call_from_clib(enqueue_wait_event.__name__, event.value)


def mark_event(event: AFEvent, /) -> None:
    """
    source: https://arrayfire.org/docs/group__event__api.htm#gab8bd1e4e5921a193d12086dda0ce8251
    """
    call_from_clib(mark_event.__name__, event.value)
