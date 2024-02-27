import random

import numpy as np
import pytest

import arrayfire_wrapper.dtypes as dtypes
from arrayfire_wrapper.lib._constants import Pad
from arrayfire_wrapper.lib.create_and_modify_array.create_array.constant import constant
from arrayfire_wrapper.lib.create_and_modify_array.create_array.pad import pad
from arrayfire_wrapper.lib.create_and_modify_array.manage_array import get_dims


@pytest.mark.parametrize(
    "original_shape",
    [
        (random.randint(1, 100),),
        (random.randint(1, 100), random.randint(1, 100)),
        (random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)),
        (random.randint(1, 100), random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)),
    ],
)
def test_zero_padding(original_shape: tuple) -> None:
    """Test if pad creates an array with no padding if no padding is given"""
    original_array = constant(2, original_shape, dtypes.s64)
    padding = Pad(0)

    zero_shape = tuple(0 for _ in range(len(original_shape)))
    result = pad(original_array, zero_shape, zero_shape, padding)

    assert get_dims(result)[0:len(original_shape)] == original_shape


@pytest.mark.parametrize(
    "original_shape",
    [
        (random.randint(1, 100),),
        (random.randint(1, 100), random.randint(1, 100)),
        (random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)),
        (random.randint(1, 100), random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)),
    ],
)
def test_negative_padding(original_shape: tuple) -> None:
    """Test if pad can properly handle if negative padding is given"""
    with pytest.raises(RuntimeError):
        original_array = constant(2, original_shape, dtypes.s64)
        padding = Pad(0)

        neg_shape = tuple(-1 for _ in range(len(original_shape)))
        result = pad(original_array, neg_shape, neg_shape, padding)

        assert get_dims(result)[0:len(original_shape)] == original_shape


@pytest.mark.parametrize(
    "original_shape",
    [
        (random.randint(1, 100),),
        (random.randint(1, 100), random.randint(1, 100)),
        (random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)),
        (random.randint(1, 100), random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)),
    ],
)
def test_padding_shape(original_shape: tuple) -> None:
    """Test if pad outputs the correct shape when a padding is adding to the original array"""
    original_array = constant(2, original_shape, dtypes.s64)
    padding = Pad(0)

    beg_shape = tuple(random.randint(1, 10) for _ in range(len(original_shape)))
    end_shape = tuple(random.randint(1, 10) for _ in range(len(original_shape)))

    result = pad(original_array, beg_shape, end_shape, padding)
    new_shape = np.array(beg_shape) + np.array(end_shape) + np.array(original_shape)

    assert get_dims(result)[0:len(original_shape)] == tuple(new_shape)
