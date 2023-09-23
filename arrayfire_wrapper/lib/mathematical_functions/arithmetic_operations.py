from __future__ import annotations

from arrayfire_wrapper.defines import AFArray
from arrayfire_wrapper.lib._utility import binary_op


def add(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    C interface wrapper to add two arrays.

    Parameters
    ----------
    lhs: AFArray
        The first input array.
    rhs: AFArray
        The second input array.

    Returns
    -------
    AFArray
        The array containing the sum of `lhs` and `rhs` after performing the operation.

    .. note::
        This function is a wrapper for the `C Interface function
    <https://arrayfire.org/docs/group__arith__func__add.htm#ga1dfbee755fedd680f4476803ddfe06a7>`_
    from the ArrayFire library.
    """
    return binary_op(add.__name__, lhs, rhs)


def bitshiftl(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__shiftl.htm#ga3139645aafe6f045a5cab454e9c13137
    """
    return binary_op(bitshiftl.__name__, lhs, rhs)


def bitshiftr(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__shiftr.htm#ga4c06b9977ecf96cdfc83b5dfd1ac4895
    """
    return binary_op(bitshiftr.__name__, lhs, rhs)


def div(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__div.htm#ga21f3f97755702692ec8976934e75fde6
    """
    return binary_op(div.__name__, lhs, rhs)


def mul(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__mul.htm#ga5f7588b2809ff7551d38b6a0bd583a02
    """
    return binary_op(mul.__name__, lhs, rhs)


def sub(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__sub.htm#ga80ff99a2e186c23614ea9f36ffc6f0a4
    """
    return binary_op(sub.__name__, lhs, rhs)
