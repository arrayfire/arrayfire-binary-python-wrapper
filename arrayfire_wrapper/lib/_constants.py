from enum import Enum


class BinaryOperator(Enum):
    ADD = 0
    MUL = 1
    MIN = 2
    MAX = 3


class ErrorCodes(Enum):
    none = 0


class Moment(Enum):
    M00 = 1
    M01 = 2
    M10 = 4
    M11 = 8
    FIRST_ORDER = M00 | M01 | M10 | M11


class PointerSource(Enum):
    device = 0  # gpu
    host = 1  # cpu


class TopK(Enum):
    DEFAULT = 0
    MIN = 1
    MAX = 2


class VarianceBias(Enum):
    DEFAULT = 0
    SAMPLE = 1
    POPULATION = 2
