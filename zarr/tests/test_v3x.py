import pytest

from zarr.v3.metadata.v3 import (
    DefaultChunkKeyConfig,
    DefaultChunkKeyEncoding,
    RegularChunkGrid,
    RegularChunkGridConfig,
)

from .test_codecs_v3 import store
from zarr.v3.array.v3 import ArrayMetadata
from zarr.v3.array.v3x import Array, lower_index


def test_v3x(store):
    zmeta = ArrayMetadata(
        shape=(10, 10),
        data_type="uint16",
        chunk_grid=RegularChunkGrid(configuration=RegularChunkGridConfig),
        chunk_key_encoding=DefaultChunkKeyEncoding(configuration=DefaultChunkKeyConfig),
        fill_value=0,
        codecs=[],
    )

    arr = Array(
        zmeta, store, shape=zmeta.shape, index=tuple(map(slice, zmeta.shape)), attributes={}
    )
    arr[slice(None), slice(None)]


# as fun as it looks
@pytest.mark.parametrize(
    "args, expected",
    (
        (((slice(0, 10, 1),), (10,)), ((((0,), (slice(0, 10, 1),))),)),
        (
            ((slice(0, 10, 1), slice(0, 10, 1)), (10, 10)),
            (((0, 0), (slice(0, 10, 1), slice(0, 10, 1))),),
        ),
        (((slice(0, 1, 1),), (10,)), (((0,), (slice(0, 1, 1),)),)),
        (
            ((slice(0, 1, 1), slice(0, 1, 1)), (10, 4)),
            (((0, 0), (slice(0, 1, 1), slice(0, 1, 1))),),
        ),
        (
            ((slice(3, 11, 1), slice(0, 1, 1)), (10, 4)),
            (
                ((0, 0), (slice(3, 10, 1), slice(0, 1, 1))),
                ((1, 0), (slice(0, 1, 1), slice(0, 1, 1))),
            ),
        ),
        (
            ((slice(3, 22, 1), slice(0, 1, 1)), (10, 4)),
            (
                ((0, 0), (slice(3, 10, 1), slice(0, 1, 1))),
                ((1, 0), (slice(0, 10, 1), slice(0, 1, 1))),
                ((2, 0), (slice(0, 2, 1), slice(0, 1, 1))),
            ),
        ),
        (
            ((slice(3, 22, 1), slice(0, 5, 1)), (20, 4)),
            (
                ((0, 0), (slice(3, 20, 1), slice(0, 4, 1))),
                ((0, 1), (slice(3, 20, 1), slice(0, 1, 1))),
                ((1, 0), (slice(0, 2, 1), slice(0, 4, 1))),
                ((1, 1), (slice(0, 2, 1), slice(0, 1, 1))),
            ),
        ),
    ),
)
def test_lower_index(args, expected):
    observed = tuple(lower_index(*args))
    assert observed == expected
