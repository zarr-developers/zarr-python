from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from zarr import create_array
from zarr.api.asynchronous import _get_shape_chunks, _like_args, group, open
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.group import AsyncGroup

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    import numpy.typing as npt

    from zarr.core.array import AsyncArray
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
    from zarr.types import AnyArray


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
    observed: AsyncArray[ArrayV2Metadata]
    | AsyncArray[ArrayV3Metadata]
    | AnyArray
    | npt.NDArray[Any],
    expected: object,
) -> None:
    """
    Test the like_args function
    """
    assert _like_args(observed) == expected


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


async def test_open_group_new_path(tmp_path: Path) -> None:
    """
    Test that zarr.api.asynchronous.group properly handles a string representation of a local file
    path that does not yet exist.
    See https://github.com/zarr-developers/zarr-python/issues/3406
    """
    # tmp_path exists, but tmp_path / "test.zarr" will not, which is important for this test
    path = tmp_path / "test.zarr"
    with pytest.warns(
        DeprecationWarning, match=re.escape("Use open_group() or create_group() instead")
    ):
        grp = await group(store=path, attributes={"a": 1})
    assert isinstance(grp, AsyncGroup)
    # Calling group on an existing store should just open that store
    with pytest.warns(
        DeprecationWarning, match=re.escape("Use open_group() or create_group() instead")
    ):
        grp = await group(store=path)
    assert grp.attrs == {"a": 1}
