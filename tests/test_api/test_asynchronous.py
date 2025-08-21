from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from zarr import create_array
from zarr.api.asynchronous import _get_shape_chunks, _like_args, open
from zarr.core.buffer.core import default_buffer_prototype

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

    from zarr.core.array import Array, AsyncArray
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata


@dataclass
class WithShape:
    shape: tuple[int, ...]


@dataclass
class WithChunks(WithShape):
    chunks: tuple[int, ...]


@dataclass
class WithChunkLen(WithShape):
    chunklen: int


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        ({}, (None, None)),
        (WithShape(shape=(1, 2)), ((1, 2), None)),
        (WithChunks(shape=(1, 2), chunks=(1, 2)), ((1, 2), (1, 2))),
        (WithChunkLen(shape=(10, 10), chunklen=1), ((10, 10), (1, 10))),
    ],
)
def test_get_shape_chunks(
    observed: object, expected: tuple[tuple[int, ...] | None, tuple[int, ...] | None]
) -> None:
    """
    Test the _get_shape_chunks function
    """
    assert _get_shape_chunks(observed) == expected


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        (np.arange(10, dtype=np.dtype("int64")), {"shape": (10,), "dtype": np.dtype("int64")}),
        (WithChunks(shape=(1, 2), chunks=(1, 2)), {"chunks": (1, 2), "shape": (1, 2)}),
        (
            create_array(
                {},
                chunks=(10,),
                shape=(100,),
                dtype="f8",
                compressors=None,
                filters=None,
                zarr_format=2,
            )._async_array,
            {
                "chunks": (10,),
                "shape": (100,),
                "dtype": np.dtype("f8"),
                "compressor": None,
                "filters": None,
                "order": "C",
            },
        ),
    ],
)
def test_like_args(
    observed: AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | Array | npt.NDArray[Any],
    expected: object,
) -> None:
    """
    Test the like_args function
    """
    assert _like_args(observed, {}) == expected


async def test_open_no_array() -> None:
    """
    Test that zarr.api.asynchronous.open attempts to open a group when no array is found, but shape was specified in kwargs.
    This behavior makes no sense but we should still test it.
    """
    store = {
        "zarr.json": default_buffer_prototype().buffer.from_bytes(
            json.dumps({"zarr_format": 3, "node_type": "group"}).encode("utf-8")
        )
    }
    with pytest.raises(
        TypeError, match=r"open_group\(\) got an unexpected keyword argument 'shape'"
    ):
        await open(store=store, shape=(1,))
