import ctypes

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._constants import Match
from arrayfire_wrapper.lib._utility import call_from_clib


def match_template(search_image: AFArray, template_image: AFArray, match_type: Match, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__cv__func__match__template.htm#ga07eaefc09c7622835baa75cc406118aa
    """
    out = AFArray.create_null_pointer()
    call_from_clib(match_template.__name__, ctypes.pointer(out), search_image, template_image, match_type.value)
    return out
