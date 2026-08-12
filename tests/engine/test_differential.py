"""The same operations through both engines must agree with numpy and each other.

The matrix covers every selection dialect (`basic`, `orthogonal`, `coordinate`,
`mask`, `block`), reading and writing, over regular and sharded arrays. numpy is
the oracle: each case is expressed as a zarr accessor call and the equivalent
numpy key, so a divergence between the engines shows up as a divergence from
numpy rather than as one engine's output being taken as the reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.errors import NegativeStepError
from zarr.storage import LocalStore

try:
    import zarrista  # noqa: F401

    ENGINES = ["default", "zarrista"]
except ImportError:
    ENGINES = ["default"]

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from zarr.abc.engine import SelectionKind
    from zarr.core.engine import EngineName

SHAPE = (10, 9)
CHUNKS = (3, 4)

MASK = np.zeros(SHAPE, dtype=bool)
MASK[0, 0] = MASK[2, 2] = MASK[4, 5] = MASK[9, 8] = True


@pytest.fixture
def reference(tmp_path: Path) -> tuple[Path, npt.NDArray[np.float64]]:
    z = zarr.create_array(LocalStore(tmp_path), shape=SHAPE, chunks=CHUNKS, dtype="float64")
    data = np.arange(90, dtype="float64").reshape(SHAPE)
    z[:, :] = data
    return tmp_path, data


@pytest.fixture
def sharded(tmp_path: Path) -> tuple[Path, npt.NDArray[np.float64]]:
    z = zarr.create_array(
        LocalStore(tmp_path),
        shape=SHAPE,
        chunks=CHUNKS,  # inner chunks
        shards=(6, 8),  # shard shape
        dtype="float64",
    )
    data = np.arange(90, dtype="float64").reshape(SHAPE)
    z[:, :] = data
    return tmp_path, data


def _read(z: zarr.Array[Any], kind: SelectionKind, selection: Any) -> npt.NDArray[Any]:
    """Read `selection` through the accessor for its dialect."""
    if kind == "basic":
        return np.asarray(z[selection])
    if kind == "orthogonal":
        return np.asarray(z.oindex[selection])
    if kind in ("coordinate", "mask"):
        return np.asarray(z.vindex[selection])
    return np.asarray(z.blocks[selection])


def _write(z: zarr.Array[Any], kind: SelectionKind, selection: Any, value: Any) -> None:
    if kind == "basic":
        z[selection] = value
    elif kind == "orthogonal":
        z.oindex[selection] = value
    elif kind in ("coordinate", "mask"):
        z.vindex[selection] = value
    else:
        z.blocks[selection] = value


def _numpy_key(kind: SelectionKind, selection: Any) -> Any:
    """The numpy key that means what zarr's `kind` selection means."""
    if kind == "orthogonal":
        # With at most one advanced index numpy's own semantics *are* outer
        # indexing; beyond that `np.ix_` is the equivalent, and a slice entry
        # has to be spelled out as the indices it covers. Cases below keep to
        # selections with no integer entry when they use more than one array,
        # since `np.ix_` cannot express a dropped axis.
        if sum(isinstance(s, np.ndarray) for s in selection) < 2:
            return selection
        return np.ix_(
            *(
                s if isinstance(s, np.ndarray) else np.arange(n)[s]
                for s, n in zip(selection, SHAPE, strict=True)
            )
        )
    if kind == "block":
        return tuple(
            slice(s * c, min((s + 1) * c, n))
            if isinstance(s, int)
            else slice((s.start or 0) * c, min(s.stop * c, n))
            for s, c, n in zip(selection, CHUNKS, SHAPE, strict=True)
        )
    return selection


READS: list[Any] = [
    pytest.param("basic", (slice(None), slice(None)), id="basic-whole"),
    pytest.param("basic", (slice(2, 7), slice(1, 5)), id="basic-box"),
    pytest.param("basic", (slice(1, 9, 2), slice(None, None, 3)), id="basic-strided"),
    pytest.param("basic", (3, slice(None)), id="basic-dropped-axis"),
    pytest.param("basic", (-1, -2), id="basic-negative-ints"),
    pytest.param("basic", (slice(4, 4), slice(None)), id="basic-empty"),
    pytest.param("basic", (Ellipsis,), id="basic-ellipsis"),
    pytest.param("orthogonal", (np.array([7, 1, 4]), np.array([0, 8])), id="oindex-arrays"),
    pytest.param("orthogonal", (3, np.array([0, 8])), id="oindex-int-and-array"),
    pytest.param("orthogonal", (np.array([7, 1, 4]), slice(1, 5)), id="oindex-array-and-slice"),
    pytest.param(
        "orthogonal",
        (np.array([True] * 5 + [False] * 5), np.array([0, 8])),
        id="oindex-bool-and-array",
    ),
    pytest.param("coordinate", (np.array([9, 0, 3]), np.array([8, 0, 2])), id="vindex-points"),
    pytest.param(
        "coordinate",
        (np.array([[0, 1], [2, 3]]), np.array([[0, 1], [2, 3]])),
        id="vindex-2d-points",
    ),
    pytest.param("mask", MASK, id="mask"),
    pytest.param("block", (1, 2), id="block-ints"),
    pytest.param("block", (slice(0, 2), 0), id="block-slice-and-int"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("kind", "selection"), READS)
def test_reads_match_numpy(
    reference: tuple[Path, npt.NDArray[np.float64]],
    engine: EngineName,
    kind: SelectionKind,
    selection: Any,
) -> None:
    tmp_path, data = reference
    z = zarr.open_array(LocalStore(tmp_path), engine=engine)
    np.testing.assert_array_equal(_read(z, kind, selection), data[_numpy_key(kind, selection)])


WRITES: list[Any] = [
    pytest.param("basic", (slice(0, 3), slice(0, 4)), 7.0, id="basic-aligned-scalar"),
    pytest.param(
        "basic",
        (slice(4, 6), slice(2, 9)),
        np.arange(14, dtype="float64").reshape(2, 7),
        id="basic-partial-chunks",
    ),
    pytest.param("basic", (slice(1, 9, 3), slice(None, None, 4)), -1.0, id="basic-strided"),
    pytest.param("basic", (9, slice(0, 4)), -2.0, id="basic-edge-chunk-row"),
    pytest.param("basic", (slice(4, 4), slice(None)), np.zeros((0, 9)), id="basic-empty"),
    pytest.param(
        "orthogonal",
        (np.array([7, 1, 4]), np.array([0, 8])),
        np.arange(6, dtype="float64").reshape(3, 2),
        id="oindex-arrays",
    ),
    pytest.param(
        "orthogonal", (3, np.array([0, 8])), np.array([11.0, 12.0]), id="oindex-int-and-array"
    ),
    pytest.param(
        "orthogonal", (np.array([7, 1, 4]), slice(1, 5)), 5.0, id="oindex-array-and-slice"
    ),
    pytest.param(
        "coordinate",
        (np.array([9, 0, 3]), np.array([8, 0, 2])),
        np.array([1.0, 2.0, 3.0]),
        id="vindex-points",
    ),
    # duplicate coordinates are legal and resolve last-wins, as they do in numpy
    pytest.param(
        "coordinate",
        (np.array([2, 2, 5]), np.array([3, 3, 6])),
        np.array([1.0, 2.0, 3.0]),
        id="vindex-duplicate-points",
    ),
    pytest.param("mask", MASK, np.array([1.0, 2.0, 3.0, 4.0]), id="mask"),
    pytest.param(
        "block", (1, 2), np.arange(3, dtype="float64").reshape(3, 1), id="block-edge-column"
    ),
    pytest.param("block", (slice(0, 2), 0), 8.0, id="block-slice-and-int"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("kind", "selection", "value"), WRITES)
def test_writes_match_numpy(
    reference: tuple[Path, npt.NDArray[np.float64]],
    engine: EngineName,
    kind: SelectionKind,
    selection: Any,
    value: Any,
) -> None:
    tmp_path, data = reference
    z = zarr.open_array(LocalStore(tmp_path), engine=engine)
    expected = data.copy()
    expected[_numpy_key(kind, selection)] = value

    _write(z, kind, selection, value)

    # the whole array, so a write that spilled outside its selection is caught
    np.testing.assert_array_equal(np.asarray(z[:, :]), expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_sharded_read_write(
    sharded: tuple[Path, npt.NDArray[np.float64]], engine: EngineName
) -> None:
    tmp_path, data = sharded
    z = zarr.open_array(LocalStore(tmp_path), engine=engine)
    np.testing.assert_array_equal(np.asarray(z[2:8, 3:9]), data[2:8, 3:9])

    expected = data.copy()
    z[1:9:2, ::3] = -5.0  # strided, so inner chunks are read-modify-written
    expected[1:9:2, ::3] = -5.0
    np.testing.assert_array_equal(np.asarray(z[:, :]), expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_scalar_read_matches_numpy(
    reference: tuple[Path, npt.NDArray[np.float64]], engine: EngineName
) -> None:
    tmp_path, data = reference
    z = zarr.open_array(LocalStore(tmp_path), engine=engine)
    assert float(np.asarray(z.get_basic_selection((3, 4)))) == data[3, 4]


@pytest.mark.parametrize(
    "selection", [(slice(None, None, -1), slice(None)), (slice(None), slice(None, None, -2))]
)
def test_negative_step_is_engine_specific(
    reference: tuple[Path, npt.NDArray[np.float64]], selection: Any
) -> None:
    """The engines deliberately differ here, so this case stays out of the
    shared matrix: zarr's own indexer refuses a negative step, while the
    zarrista engine resolves the raw selection through `zarr_indexing` and
    serves it.
    """
    tmp_path, data = reference
    with pytest.raises(NegativeStepError):
        zarr.open_array(LocalStore(tmp_path), engine="default")[selection]

    if "zarrista" not in ENGINES:
        pytest.skip("zarrista not installed")
    z = zarr.open_array(LocalStore(tmp_path), engine="zarrista")
    np.testing.assert_array_equal(np.asarray(z[selection]), data[selection])

    expected = data.copy()
    expected[selection] = -data[selection]
    z[selection] = -data[selection]
    np.testing.assert_array_equal(np.asarray(z[:, :]), expected)
