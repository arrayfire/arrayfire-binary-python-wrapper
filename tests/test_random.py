import random

import pytest

import arrayfire_wrapper.dtypes as dtypes
import arrayfire_wrapper.lib as wrapper

invalid_shape = (
    random.randint(1, 10),
    random.randint(1, 10),
    random.randint(1, 10),
    random.randint(1, 10),
    random.randint(1, 10),
)


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
def test_randu_shape(shape: tuple) -> None:
    """Test if randu function creates an array with the correct shape."""
    dtype = dtypes.s16

    result = wrapper.randu(shape, dtype)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203


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
def test_random_uniform_shape(shape: tuple) -> None:
    """Test if rand uniform function creates an array with the correct shape."""
    dtype = dtypes.s16
    engine = wrapper.create_random_engine(100, 10)

    result = wrapper.random_uniform(shape, dtype, engine)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203


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
def test_randn_shape(shape: tuple) -> None:
    """Test if randn function creates an array with the correct shape."""
    dtype = dtypes.f32

    result = wrapper.randn(shape, dtype)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203


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
def test_random_normal_shape(shape: tuple) -> None:
    """Test if random normal function creates an array with the correct shape."""
    dtype = dtypes.f32
    engine = wrapper.create_random_engine(100, 10)

    result = wrapper.random_normal(shape, dtype, engine)

    assert wrapper.get_dims(result)[0 : len(shape)] == shape  # noqa: E203


@pytest.mark.parametrize(
    "engine_index",
    [100, 200, 300],
)
def test_create_random_engine(engine_index: int) -> None:
    engine = wrapper.create_random_engine(engine_index, 10)

    engine_type = wrapper.random_engine_get_type(engine)

    assert engine_type == engine_index


@pytest.mark.parametrize(
    "invalid_index",
    [random.randint(301, 600), random.randint(301, 600), random.randint(301, 600)],
)
def test_invalid_random_engine(invalid_index: int) -> None:
    "Test if invalid engine types are properly handled"
    with pytest.raises(RuntimeError):

        invalid_engine = wrapper.create_random_engine(invalid_index, 10)

        engine_type = wrapper.random_engine_get_type(invalid_engine)

        assert engine_type == invalid_engine
