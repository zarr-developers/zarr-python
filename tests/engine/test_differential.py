"""The same operations through both engines must agree with numpy and each other."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import zarr
from zarr.storage import LocalStore

try:
    import zarrista  # noqa: F401

    ENGINES = ["default", "zarrista"]
except ImportError:
    ENGINES = ["default"]

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from zarr.core.engine import EngineName

SHAPE = (10, 9)
CHUNKS = (3, 4)


@pytest.fixture
def reference(tmp_path: Path) -> tuple[Path, npt.NDArray[np.float64]]:
    z = zarr.create_array(LocalStore(tmp_path), shape=SHAPE, chunks=CHUNKS, dtype="float64")
    data = np.arange(90, dtype="float64").reshape(SHAPE)
    z[:, :] = data
    return tmp_path, data


READS: list[tuple[int | slice, int | slice]] = [
    (slice(None), slice(None)),
    (slice(2, 7), slice(1, 5)),
    # negative-step slices are rejected by `Array.__getitem__` (see
    # `NegativeStepError`); only positive step > 1 is supported, so this
    # covers strided reads on both axes without hitting that restriction.
    (slice(1, 9, 2), slice(None, None, 3)),
    (3, slice(None)),
    (-1, -2),
    (slice(4, 4), slice(None)),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("sel", READS)
def test_reads_match_numpy(
    reference: tuple[Path, npt.NDArray[np.float64]],
    engine: EngineName,
    sel: tuple[int | slice, int | slice],
) -> None:
    tmp_path, data = reference
    z = zarr.open_array(LocalStore(tmp_path), engine=engine)
    np.testing.assert_array_equal(np.asarray(z[sel]), data[sel])


@pytest.mark.parametrize("engine", ENGINES)
def test_fancy_reads_match_numpy(
    reference: tuple[Path, npt.NDArray[np.float64]], engine: EngineName
) -> None:
    tmp_path, data = reference
    z = zarr.open_array(LocalStore(tmp_path), engine=engine)
    np.testing.assert_array_equal(
        np.asarray(z.oindex[np.array([7, 1, 4]), np.array([0, 8])]),
        data[np.ix_([7, 1, 4], [0, 8])],
    )
    np.testing.assert_array_equal(
        np.asarray(z.vindex[np.array([9, 0, 3]), np.array([8, 0, 2])]),
        data[np.array([9, 0, 3]), np.array([8, 0, 2])],
    )
    np.testing.assert_array_equal(np.asarray(z.blocks[1, 2]), data[3:6, 8:9])


@pytest.mark.parametrize("engine", ENGINES)
def test_writes_match_numpy(
    reference: tuple[Path, npt.NDArray[np.float64]], engine: EngineName
) -> None:
    tmp_path, data = reference
    z = zarr.open_array(LocalStore(tmp_path), engine=engine)
    expected = data.copy()

    z[0:3, 0:4] = 7.0  # aligned full chunk
    expected[0:3, 0:4] = 7.0
    z[4:6, 2:9] = np.arange(14, dtype="float64").reshape(2, 7)  # partial chunks
    expected[4:6, 2:9] = np.arange(14, dtype="float64").reshape(2, 7)
    z[1:9:3, ::4] = -1.0  # strided write (facade RMW)
    expected[1:9:3, ::4] = -1.0

    np.testing.assert_array_equal(np.asarray(z[:, :]), expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_sharded_reads(
    reference: tuple[Path, npt.NDArray[np.float64]],
    engine: EngineName,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    path = tmp_path_factory.mktemp("sharded")
    z = zarr.create_array(
        LocalStore(path),
        shape=SHAPE,
        chunks=(3, 4),  # inner chunks
        shards=(6, 8),  # shard shape
        dtype="int32",
    )
    data = np.arange(90, dtype="int32").reshape(SHAPE)
    z[:, :] = data
    zr = zarr.open_array(LocalStore(path), engine=engine)
    np.testing.assert_array_equal(np.asarray(zr[2:8, 3:9]), data[2:8, 3:9])
