from enum import Enum


class BinaryOperator(Enum):  # Binary Operators
    ADD = 0
    MUL = 1
    MIN = 2
    MAX = 3


class CannyThreshold(Enum):  # Canny Edge Threshold Types
    MANUAL = 0
    AUTO_OTSU = 1


class Connectivity(Enum):  # Neighbourhood connectivity
    FOUR = 4
    EIGHT = 8


class ConvGradient(Enum):  # Convolution Gradient Types
    DEFAULT = 0
    FILTER = 1
    DATA = 2
    BIAS = 3


class CSpace(Enum):  # Colorspace formats
    GRAY = 0
    RGB = 1
    HSV = 2
    YCbCr = 3


class Diffusion(Enum):  # Diffusion equations
    DEFAULT = 0
    GRAD = 1
    MCDE = 2


class ErrorCodes(Enum):  # Error Values
    NONE = 0

    # 100-199 Errors in environment
    NO_MEM = 101
    DRIVER = 102
    RUNTIME = 103

    # 200-299 Errors in input parameters
    INVALID_ARRAY = 201
    ARG = 202
    SIZE = 203
    TYPE = 204
    DIFF_TYPE = 205
    BATCH = 207
    DEVICE = 208

    # 300-399 Errors for missing software features
    NOT_SUPPORTED = 301
    NOT_CONFIGURED = 302
    NONFREE = 303

    # 400-499 Errors for missing hardware features
    NO_DBL = 401
    NO_GFX = 402
    NO_HALF = 403

    # 500-599 Errors specific to the heterogeneous API
    LOAD_LIB = 501
    LOAD_SYM = 502
    ARR_BKND_MISMATCH = 503

    # 900-999 Errors from upstream libraries and runtimes
    INTERNAL = 998
    UNKNOWN = 999


class Flux(Enum):
    DEFAULT = 0
    QUADRATIC = 1
    EXPONENTIAL = 2


class Interp(Enum):  # Interpolation method types
    NEAREST = 0
    LINEAR = 1
    BILINEAR = 2
    CUBIC = 3
    LOWER = 4
    LINEAR_COSINE = 5
    BILINEAR_COSINE = 6
    BICUBIC = 7
    CUBIC_SPLINE = 8
    BICUBIC_SPLINE = 9


class IterativeDeconv(Enum):  # Iterative deconvolution algorithm
    DEFAULT = 0
    LANDWEBER = 1
    RICHARDSONLUCY = 2


class Match(Enum):
    SAD = 0  # Sum of absolute differences
    ZSAD = 1  # Zero mean SAD
    LSAD = 2  # Locally scaled SAD
    SSD = 3  # Sum of squared differences
    ZSSD = 4  # Zero mean SSD
    LSSD = 5  # Locally scaled SSD
    NCC = 6  # Normalized cross correlation
    ZNCC = 7  # Zero mean NCC
    SHD = 8  # Sum of hamming distances


class Moment(Enum):  # Image moments types
    M00 = 1
    M01 = 2
    M10 = 4
    M11 = 8
    FIRST_ORDER = M00 | M01 | M10 | M11


class Pad(Enum):  # Edge padding types
    ZERO = 0
    SYM = 1
    CLAMP_TO_EDGE = 2
    PERIODIC = 3


class PointerSource(Enum):
    device = 0  # gpu
    host = 1  # cpu


class TopK(Enum):  # Top-K ordering
    DEFAULT = 0
    MIN = 1
    MAX = 2


class VarianceBias(Enum):  # Variance Bias types
    DEFAULT = 0
    SAMPLE = 1
    POPULATION = 2


class YCCStd(Enum):  # YCC Standard formats
    YCC_601 = 601
    YCC_709 = 709
    YCC_2020 = 2020
