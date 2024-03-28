import random

import pytest

import arrayfire_wrapper.dtypes as dtypes
import arrayfire_wrapper.lib as wrapper


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10), 1),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
def test_range_shape(shape: tuple) -> None:
    """Test if the range function output an AFArray with the correct shape"""
    dim = 2
    dtype = dtypes.s16

    result = wrapper.range(shape, dim, dtype)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203


def test_range_invalid_shape() -> None:
    """Test if range function correctly handles an invalid shape"""
    with pytest.raises(TypeError):
        shape = (
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
        )
        dim = 2
        dtype = dtypes.s16

        wrapper.range(shape, dim, dtype)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (random.randint(1, 10), 1),
        (random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)),
    ],
)
def test_range_invalid_dim(shape: tuple) -> None:
    """Test if the range function can properly handle and invalid dimension given"""
    with pytest.raises(RuntimeError):
        dim = random.randint(4, 10)
        dtype = dtypes.s16

        wrapper.range(shape, dim, dtype)
