from __future__ import annotations

import ctypes
import math
from typing import Any

from arrayfire_wrapper.lib._broadcast import bcast_var
from arrayfire_wrapper.lib.create_and_modify_array.manage_array import release_array, retain_array
from arrayfire_wrapper.defines import AFArray

class _IndexSequence(ctypes.Structure):
    """
    arrayfire equivalent of slice

    Attributes
    ----------

    begin: number
           Start of the sequence.

    end  : number
           End of sequence.

    step : number
           Step size.

    Parameters
    ----------

    chunk: slice or number.

    """

    # More about _fields_ purpose: https://docs.python.org/3/library/ctypes.html#structures-and-unions
    _fields_ = [
        ("begin", ctypes.c_double),
        ("end", ctypes.c_double),
        ("step", ctypes.c_double),
    ]

    def __init__(self, chunk: int | slice):
        self.begin = ctypes.c_double(0)
        self.end = ctypes.c_double(-1)
        self.step = ctypes.c_double(1)

        if isinstance(chunk, int):
            self.begin = ctypes.c_double(chunk)
            self.end = ctypes.c_double(chunk)

        elif isinstance(chunk, slice):
            if chunk.step:
                self.step = ctypes.c_double(chunk.step)
                if chunk.step < 0:
                    self.begin, self.end = self.end, self.begin

            if chunk.start:
                self.begin = ctypes.c_double(chunk.start)

            if chunk.stop:
                self.end = ctypes.c_double(chunk.stop)

            # handle special cases
            if 0 <= self.end <= self.begin and self.step >= 0:  # type: ignore[operator]
                self.begin.value = 1
                self.end.value = 1
                self.step.value = 1

            elif self.begin <= self.end < 0 and self.step <= 0:  # type: ignore[operator]
                self.begin.value = -2
                self.end.value = -2
                self.step.value = -1

            if chunk.stop:
                self.end -= math.copysign(1, self.step)  # type: ignore[operator, assignment, arg-type]  # FIXME
        else:
            raise IndexError("Invalid type while indexing arrayfire.array")


class ParallelRange(_IndexSequence):
    """
    Class used to parallelize for loop.

    Inherits from _IndexSequence.

    Attributes
    ----------

    chunk: slice

    Parameters
    ----------

    start: number
           Beginning of parallel range.

    stop : number
           End of parallel range.

    step : number
           Step size for parallel range.

    Examples
    --------

    >>> import arrayfire as af
    >>> a = af.randu(3, 3)
    >>> b = af.randu(3, 1)
    >>> c = af.constant(0, 3, 3)
    >>> for ii in af.ParallelRange(3):
    ...     c[:, ii] = a[:, ii] + b
    ...
    >>> af.display(a)
    [3 3 1 1]
        0.4107     0.1794     0.3775
        0.8224     0.4198     0.3027
        0.9518     0.0081     0.6456

    >>> af.display(b)
    [3 1 1 1]
        0.7269
        0.7104
        0.5201

    >>> af.display(c)
    [3 3 1 1]
        1.1377     0.9063     1.1045
        1.5328     1.1302     1.0131
        1.4719     0.5282     1.1657

    """

    def __init__(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float | None = None,
    ) -> None:
        if not stop:
            stop = start
            start = 0

        self.chunk = slice(start, stop, step)
        super().__init__(self.chunk)

    def __iter__(self) -> ParallelRange:
        return self

    def __next__(self) -> ParallelRange:
        if bcast_var.get() is True:
            bcast_var.toggle()
            raise StopIteration
        else:
            bcast_var.toggle()
            return self


class _IndexUnion(ctypes.Union):
    _fields_ = [("arr", ctypes.c_void_p), ("seq", _IndexSequence)]


class IndexStructure(ctypes.Structure):
    _fields_ = [
        ("idx", _IndexUnion),
        ("isSeq", ctypes.c_bool),
        ("isBatch", ctypes.c_bool),
    ]

    """
    Container for the index class in arrayfire C library

    Attributes
    ----------
    idx.arr: ctypes.c_void_p
             - Default 0

    idx.seq: af._IndexSequence
             - Default af._IndexSequence(0, -1, 1)

    isSeq   : bool
            - Default True

    isBatch : bool
            - Default False

    Parameters
    -----------

    idx: key
         - If of type AFArray, self.idx.arr = idx, self.isSeq = False
         - If of type af.ParallelRange, self.idx.seq = idx, self.isBatch = True
         - Default:, self.idx.seq = af._IndexSequence(idx)

    Note
    ----

    Implemented for internal use only. Use with extreme caution.

    """

    def __init__(self, idx: int | slice | AFArray) -> None:
        self.idx = _IndexUnion()
        self.isBatch = False
        self.isSeq = True

        if isinstance(idx, int) or isinstance(idx, slice):
            self.idx.seq = _IndexSequence(idx)
        elif isinstance(idx, ParallelRange):
            self.idx.seq = idx
            self.isBatch = True
        elif isinstance(idx, AFArray):
            self.idx.arr = retain_array(idx)
            self.isSeq = False
        else:
            raise IndexError("Invalid type while indexing arrayfire.array")

    def __del__(self) -> None:
        if not self.isSeq:
            # ctypes field variables are automatically
            # converted to basic C types so we have to
            # build the void_p from the value again.
            arr = ctypes.c_void_p(self.idx.arr)
            release_array(arr)  # type: ignore[arg-type]


class CIndexStructure:
    def __init__(self) -> None:
        index_vec = IndexStructure * 4
        # NOTE Do not lose those idx as self.array keeps no reference to them. Otherwise the destructor
        # is prematurely called
        self.idxs = [IndexStructure(slice(None))] * 4
        self.array = index_vec(*self.idxs)

    @property
    def pointer(self) -> ctypes._Pointer:
        return ctypes.pointer(self.array)

    def __getitem__(self, idx: int) -> IndexStructure:
        return self.array[idx]

    def __setitem__(self, idx: int, value: IndexStructure) -> None:
        self.array[idx] = value
        self.idxs[idx] = value


def get_indices(key: int | slice | tuple[int | slice | AFArray, ...] | AFArray) -> CIndexStructure:  # BUG
    indices = CIndexStructure()

    if isinstance(key, tuple):
        for n in range(len(key)):
            indices[n] = IndexStructure(key[n])
    else:
        indices[0] = IndexStructure(key)

    return indices
