"""End-to-end tests for the public lazy-indexing API (``Array.lazy``).

These exercise lazy views — ``arr.lazy[sel]``, ``arr.lazy.oindex[sel]``,
``arr.lazy.vindex[sel]`` — and their write-through and composition behaviour,
in every case comparing against an equivalent NumPy reference.

Coverage is parametrized over a matrix of array geometries so the same dense
set of selections runs against 0-D arrays, and 1/2/3-D arrays in both
**unsharded** and **sharded** form. Sharded arrays route through inner-chunk
geometry (partial-shard read-modify-write), so running the full matrix against
them guards the transform read/write path where it is most likely to break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import MemoryStore


@dataclass(frozen=True)
class Config:
    """A single array geometry: shape, inner chunk shape, optional shard shape."""

    id: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None


CONFIGS = [
    Config("0d", (), (), None),
    Config("1d-unsharded", (24,), (4,), None),
    Config("1d-sharded", (24,), (4,), (12,)),
    Config("2d-unsharded", (20, 30), (5, 10), None),
    Config("2d-sharded", (20, 30), (5, 10), (10, 30)),
    Config("3d-unsharded", (8, 6, 10), (2, 3, 5), None),
    Config("3d-sharded", (8, 6, 10), (2, 3, 5), (4, 6, 10)),
]

# Basic (slice/int/ellipsis) selections valid for each rank. Includes
# across-boundary and positive-strided cases. Negative steps are unsupported
# by zarr (see TestLazyErrors.test_negative_step_raises), so they are excluded.
BASIC_SELECTIONS: dict[int, list[Any]] = {
    0: [Ellipsis],
    1: [slice(3, 20), 7, Ellipsis, slice(None, None, 2), slice(5, 22)],
    2: [
        (slice(2, 8), slice(5, 15)),
        3,
        (3, 5),
        Ellipsis,
        (slice(None, None, 2), slice(None, None, 3)),
        (slice(3, 17), slice(8, 22)),
        (slice(None), 5),
    ],
    3: [
        (slice(1, 7), slice(0, 4), slice(2, 9)),
        2,
        (1, 2, 3),
        Ellipsis,
        (slice(None, None, 2), slice(None, None, 2), slice(None, None, 2)),
        (slice(2, 7), slice(1, 5), slice(3, 9)),
    ],
}

# Per-rank integer index arrays for orthogonal / vectorized fancy indexing.
FANCY_INDICES: dict[int, tuple[npt.NDArray[np.intp], ...]] = {
    1: (np.array([1, 3, 7, 20], dtype=np.intp),),
    2: (np.array([1, 5, 10, 18], dtype=np.intp), np.array([0, 3, 9, 29], dtype=np.intp)),
    3: (
        np.array([1, 5, 7], dtype=np.intp),
        np.array([0, 3, 5], dtype=np.intp),
        np.array([2, 7, 9], dtype=np.intp),
    ),
}

ND_CONFIGS = [c for c in CONFIGS if len(c.shape) >= 1]


def _make(cfg: Config) -> tuple[zarr.Array[Any], npt.NDArray[Any]]:
    """Build a zarr array for ``cfg`` and an identical NumPy reference."""
    a = zarr.create_array(
        MemoryStore(), shape=cfg.shape, chunks=cfg.chunks, shards=cfg.shards, dtype="i4"
    )
    n = int(np.prod(cfg.shape, dtype=int))  # prod(()) == 1 -> a single 0-D element
    ref = np.arange(n, dtype="i4").reshape(cfg.shape)
    a[...] = ref
    return a, ref


def _value_like(expected: npt.NDArray[Any]) -> Any:
    """A distinctly-valued array (or scalar) shaped like ``expected`` for writes."""
    shape = np.shape(expected)
    val = (np.arange(int(np.prod(shape, dtype=int)), dtype="i4").reshape(shape) + 1) * 7 + 1
    return val[()] if shape == () else val  # 0-D -> python/np scalar


def _fmt(sel: Any) -> str:
    if sel is Ellipsis:
        return "..."
    if isinstance(sel, tuple):
        return ",".join(_fmt(s) for s in sel)
    if isinstance(sel, slice):
        start = "" if sel.start is None else str(sel.start)
        stop = "" if sel.stop is None else str(sel.stop)
        step = f":{sel.step}" if sel.step is not None else ""
        return f"{start}:{stop}{step}"
    if isinstance(sel, np.ndarray):
        return "x".join(map(str, sel.shape))
    return str(sel)


BASIC_CASES = [
    pytest.param(cfg, sel, id=f"{cfg.id}:{_fmt(sel)}")
    for cfg in CONFIGS
    for sel in BASIC_SELECTIONS[len(cfg.shape)]
]
ND_CASES = [pytest.param(cfg, id=cfg.id) for cfg in ND_CONFIGS]
# Configs with >= 2 dimensions, i.e. enough to exercise orthogonal indexing
# across multiple array axes (the outer-product case).
MULTI_AXIS = [cfg for cfg in ND_CONFIGS if len(cfg.shape) >= 2]
MULTI_AXIS_CASES = [pytest.param(cfg, id=cfg.id) for cfg in MULTI_AXIS]
MULTI_AXIS_UNSHARDED_CASES = [
    pytest.param(cfg, id=cfg.id) for cfg in MULTI_AXIS if cfg.shards is None
]
MULTI_AXIS_SHARDED_CASES = [
    pytest.param(cfg, id=cfg.id) for cfg in MULTI_AXIS if cfg.shards is not None
]


def _oindex_one_axis(cfg: Config) -> tuple[Any, ...]:
    """An orthogonal selection with a single fancy axis (axis 0) and slices elsewhere."""
    idx = FANCY_INDICES[len(cfg.shape)][0]
    return (idx, *([slice(None)] * (len(cfg.shape) - 1)))


def _rand_slice(rng: np.random.Generator, size: int) -> slice:
    """A random non-empty positive-step slice within ``[0, size)``."""
    start = int(rng.integers(0, size))
    stop = int(rng.integers(start + 1, size + 1))
    step = int(rng.integers(1, 4))
    return slice(start, stop, step)


def _unique_coords(
    rng: np.random.Generator, shape: tuple[int, ...], n: int
) -> tuple[npt.NDArray[np.intp], ...]:
    """``n`` distinct coordinate tuples (no duplicate points → write order irrelevant)."""
    total = int(np.prod(shape, dtype=int))
    flat = rng.choice(total, size=min(n, total), replace=False)
    return tuple(c.astype(np.intp) for c in np.unravel_index(flat, shape))


# Multi-dim geometries used for the randomized model-based round-trips (the 0-D /
# 1-D configs have too small a selection space to be interesting here).
RANDOM_CASES = [pytest.param(cfg, id=cfg.id) for cfg in CONFIGS if len(cfg.shape) >= 2]


class TestLazyBasicRead:
    @pytest.mark.parametrize(("cfg", "sel"), BASIC_CASES)
    def test_matches_numpy(self, cfg: Config, sel: Any) -> None:
        """A lazy basic view reads identically to NumPy (and to eager indexing)."""
        a, ref = _make(cfg)
        expected = ref[sel]
        view = a.lazy[sel]
        assert tuple(view.shape) == np.shape(expected)
        np.testing.assert_array_equal(view[...], expected)
        np.testing.assert_array_equal(a[sel], expected)  # eager parity, all geometries

    @pytest.mark.parametrize(("cfg", "sel"), BASIC_CASES)
    def test_result_and_asarray(self, cfg: Config, sel: Any) -> None:
        """``view.result()`` and ``np.asarray(view)`` agree with direct resolution."""
        a, ref = _make(cfg)
        expected = ref[sel]
        view = a.lazy[sel]
        np.testing.assert_array_equal(view.result(), expected)
        np.testing.assert_array_equal(np.asarray(view), np.asarray(expected))


class TestLazyBasicWrite:
    @pytest.mark.parametrize(("cfg", "sel"), BASIC_CASES)
    def test_write_through(self, cfg: Config, sel: Any) -> None:
        """Assigning through a lazy basic view mutates exactly the selected region."""
        a, ref = _make(cfg)
        expected = ref.copy()
        val = _value_like(ref[sel])
        expected[sel] = val
        a.lazy[sel] = val
        np.testing.assert_array_equal(a[...], expected)


class TestLazyOIndex:
    @pytest.mark.parametrize("cfg", ND_CASES)
    def test_read_single_axis(self, cfg: Config) -> None:
        """Lazy orthogonal indexing on a single fancy axis matches NumPy."""
        a, ref = _make(cfg)
        sel = _oindex_one_axis(cfg)
        expected = ref[sel]
        view = a.lazy.oindex[sel]
        assert tuple(view.shape) == expected.shape
        np.testing.assert_array_equal(view[...], expected)

    @pytest.mark.parametrize("cfg", ND_CASES)
    def test_write_single_axis(self, cfg: Config) -> None:
        """Write-through lazy orthogonal indexing on a single fancy axis updates the region."""
        a, ref = _make(cfg)
        sel = _oindex_one_axis(cfg)
        expected = ref.copy()
        val = _value_like(ref[sel])
        expected[sel] = val
        a.lazy.oindex[sel] = val
        np.testing.assert_array_equal(a[...], expected)

    @pytest.mark.parametrize("cfg", MULTI_AXIS_CASES)
    def test_multi_axis_read(self, cfg: Config) -> None:
        """Lazy orthogonal indexing across >=2 array axes is the outer product (np.ix_)."""
        a, ref = _make(cfg)
        idx = FANCY_INDICES[len(cfg.shape)]
        expected = ref[np.ix_(*idx)]
        view = a.lazy.oindex[idx]
        assert tuple(view.shape) == expected.shape
        np.testing.assert_array_equal(view[...], expected)

    @pytest.mark.parametrize("cfg", MULTI_AXIS_UNSHARDED_CASES)
    def test_multi_axis_write(self, cfg: Config) -> None:
        """Write-through lazy orthogonal indexing across >=2 array axes (unsharded)."""
        a, ref = _make(cfg)
        idx = FANCY_INDICES[len(cfg.shape)]
        expected = ref.copy()
        val = _value_like(ref[np.ix_(*idx)])
        expected[np.ix_(*idx)] = val
        a.lazy.oindex[idx] = val
        np.testing.assert_array_equal(a[...], expected)

    @pytest.mark.xfail(
        strict=True,
        reason="orthogonal multi-array writes are unsupported by the sharding "
        "partial-write codec (eager oindex raises the same way); a strict xfail "
        "flags if sharded support is added",
    )
    @pytest.mark.parametrize("cfg", MULTI_AXIS_SHARDED_CASES)
    def test_multi_axis_write_sharded_unsupported(self, cfg: Config) -> None:
        """Sharded orthogonal multi-array write — a pre-existing codec limitation."""
        a, ref = _make(cfg)
        idx = FANCY_INDICES[len(cfg.shape)]
        a.lazy.oindex[idx] = _value_like(ref[np.ix_(*idx)])


class TestLazyVIndex:
    @pytest.mark.parametrize("cfg", ND_CASES)
    def test_read(self, cfg: Config) -> None:
        """Lazy vectorized indexing matches NumPy's point (coordinate) selection."""
        a, ref = _make(cfg)
        idx = FANCY_INDICES[len(cfg.shape)]
        expected = ref[idx]
        view = a.lazy.vindex[idx]
        assert tuple(view.shape) == expected.shape
        np.testing.assert_array_equal(view[...], expected)

    @pytest.mark.parametrize("cfg", ND_CASES)
    def test_write(self, cfg: Config) -> None:
        """Write-through lazy vectorized indexing updates the selected points."""
        a, ref = _make(cfg)
        idx = FANCY_INDICES[len(cfg.shape)]
        expected = ref.copy()
        val = _value_like(ref[idx])
        expected[idx] = val
        a.lazy.vindex[idx] = val
        np.testing.assert_array_equal(a[...], expected)


class TestLazyComposition:
    @pytest.mark.parametrize("cfg", ND_CASES)
    def test_chained_views_compose(self, cfg: Config) -> None:
        """Composing two lazy slices equals applying them in sequence on NumPy."""
        a, ref = _make(cfg)
        view = a.lazy[1:-1].lazy[1:-1]
        expected = ref[1:-1][1:-1]
        assert tuple(view.shape) == expected.shape
        np.testing.assert_array_equal(view[...], expected)


class TestLazyViewMethods:
    """Indexing methods called on a lazy *view* object (``v = arr.lazy[...]``) must
    honor the view's transform, not silently fall back to the storage grid.

    The accessor (``arr.lazy.oindex[...]``) was covered elsewhere, but the methods
    on the *returned* non-identity Array (``v.oindex[...]``, ``v[..., -1]``) route
    through ``Array``'s own dispatch, which had correctness gaps.
    """

    @pytest.mark.parametrize("cfg", ND_CASES)
    def test_view_oindex_respects_transform(self, cfg: Config) -> None:
        """``v.oindex[idx]`` on a sub-view reads from the view, not the base array."""
        a, ref = _make(cfg)
        n0 = cfg.shape[0]
        cut = n0 // 2
        vslice = (slice(cut, n0), *([slice(None)] * (len(cfg.shape) - 1)))
        vref = ref[vslice]
        v = a.lazy[vslice]
        idx = np.array([0, vref.shape[0] - 1], dtype=np.intp)
        osel: Any = (idx, *([slice(None)] * (len(cfg.shape) - 1)))
        np.testing.assert_array_equal(v.oindex[osel], vref[osel])

    @pytest.mark.parametrize("cfg", MULTI_AXIS_CASES)
    def test_view_negative_index_after_ellipsis(self, cfg: Config) -> None:
        """``v[..., -1]`` on a view selects the last element of the last axis."""
        a, ref = _make(cfg)
        v = a.lazy[1 : cfg.shape[0]]
        vref = ref[1 : cfg.shape[0]]
        np.testing.assert_array_equal(v[..., -1], vref[..., -1])

    @pytest.mark.parametrize("cfg", MULTI_AXIS_CASES)
    def test_view_basic_tuple_and_int(self, cfg: Config) -> None:
        """Basic tuple selections on a view (incl. integer axes that drop) honor the view."""
        a, ref = _make(cfg)
        cut = cfg.shape[0] // 2
        v = a.lazy[cut:]
        vref = ref[cut:]
        full: Any = (slice(0, 2), *([slice(None)] * (len(cfg.shape) - 1)))
        np.testing.assert_array_equal(v[full], vref[full])
        intsel: Any = (1, *([slice(None)] * (len(cfg.shape) - 1)))
        np.testing.assert_array_equal(v[intsel], vref[intsel])  # int axis drops

    @pytest.mark.parametrize("cfg", MULTI_AXIS_CASES)
    def test_view_vindex(self, cfg: Config) -> None:
        """``v.vindex[...]`` (coordinate selection) on a view honors the view."""
        a, ref = _make(cfg)
        cut = cfg.shape[0] // 2
        v = a.lazy[cut:]
        vref = ref[cut:]
        idx = tuple(np.array([0, 1, 2], dtype=np.intp) for _ in cfg.shape)
        np.testing.assert_array_equal(v.vindex[idx], vref[idx])

    def test_view_vindex_with_flat_out_buffer(self) -> None:
        """vindex with a multi-dim result and out= on a view uses a flat out buffer.

        Vectorized indexing scatters through a single flat index, so (as in the
        eager path) the out buffer must be flat (shape = number of points).
        """
        a, ref = _make(CONFIGS[3])  # 2d-unsharded
        v = a.lazy[2:18]
        vref = ref[2:18]
        i0 = np.array([[0, 1], [2, 3]], dtype=np.intp)
        i1 = np.array([[0, 5], [10, 15]], dtype=np.intp)
        expected = vref[i0, i1]
        buf = default_buffer_prototype().nd_buffer.empty(
            shape=(expected.size,), dtype=np.dtype("i4")
        )
        v.get_coordinate_selection((i0, i1), out=buf)
        np.testing.assert_array_equal(
            np.asarray(buf.as_ndarray_like()).reshape(expected.shape), expected
        )

    @pytest.mark.parametrize("cfg", MULTI_AXIS_UNSHARDED_CASES)
    def test_view_write_through_tuple(self, cfg: Config) -> None:
        """Writing through a view with a basic tuple selection lands in view coords."""
        a, ref = _make(cfg)
        cut = cfg.shape[0] // 2
        v = a.lazy[cut:]
        expected = ref.copy()
        wsel: Any = (slice(0, 2), *([slice(None)] * (len(cfg.shape) - 1)))
        val = _value_like(ref[cut:][wsel])
        expected[cut:][wsel] = val
        v[wsel] = val
        np.testing.assert_array_equal(a[...], expected)


class TestLazyErrors:
    def test_negative_step_raises(self) -> None:
        """A negative slice step is unsupported and raises on the lazy path."""
        a, _ = _make(CONFIGS[1])  # 1d-unsharded
        with pytest.raises(IndexError, match="step must be positive"):
            a.lazy[::-1]

    def test_accessor_negative_index_is_literal(self) -> None:
        """The lazy accessor copies TensorStore: indices are literal coordinates.

        Negative indices are absolute (origin 0), not from-the-end, so they fall
        outside the ``[0, N)`` domain and raise — unlike the NumPy-normalizing
        eager/view-method paths. Out-of-bounds array values raise cleanly rather
        than silently wrapping.
        """
        a, _ = _make(CONFIGS[1])  # 1d-unsharded, shape (24,)
        with pytest.raises(IndexError):
            _ = a.lazy[-1][...]
        for sel in (np.array([-1], dtype=np.intp), np.array([24], dtype=np.intp)):
            with pytest.raises(IndexError, match="out of bounds"):
                _ = a.lazy.oindex[(sel,)][...]
            with pytest.raises(IndexError, match="out of bounds"):
                _ = a.lazy.vindex[(sel,)][...]

    def test_block_selection_on_view_rejected(self) -> None:
        """Block selection is ill-defined on a lazy view and must raise, not corrupt."""
        a, _ = _make(CONFIGS[3])  # 2d-unsharded
        v = a.lazy[5:15]
        with pytest.raises(NotImplementedError, match="block selection"):
            _ = v.blocks[0]
        with pytest.raises(NotImplementedError, match="block selection"):
            v.blocks[0] = 0

    def test_vindex_with_slice_rejected(self) -> None:
        """vindex is coordinate-only; mixing a slice raises (parity with eager)."""
        a, _ = _make(CONFIGS[3])  # 2d-unsharded
        with pytest.raises(IndexError):
            a.lazy.vindex[(np.array([0, 1], dtype=np.intp), slice(None))]

    def test_result_threads_prototype(self) -> None:
        """``result(prototype=...)`` forwards the prototype rather than dropping it."""
        a, ref = _make(CONFIGS[3])  # 2d-unsharded
        proto = default_buffer_prototype()
        with mock.patch.object(type(a), "get_basic_selection", autospec=True) as gbs:
            gbs.return_value = ref
            a.result(prototype=proto)
        assert gbs.call_args.kwargs.get("prototype") is proto


class TestLazyRandomizedRoundtrip:
    """Model-based round-trips: apply random selections to both the zarr array and a
    NumPy reference, then assert they stay equal. Parametrized over sharded and
    unsharded grids with one body, so the same selections exercise chunk- and
    shard-boundary read-modify-write (the pattern TensorStore's driver_testutil uses).
    """

    @pytest.mark.parametrize("cfg", RANDOM_CASES)
    def test_basic_writes_track_numpy(self, cfg: Config) -> None:
        """Random strided basic writes/reads match a NumPy model across boundaries."""
        rng = np.random.default_rng(0)
        a, ref = _make(cfg)
        ref = ref.copy()
        for _ in range(25):
            sel = tuple(_rand_slice(rng, s) for s in cfg.shape)
            val = rng.integers(-9999, 9999, size=ref[sel].shape, dtype="i4")
            ref[sel] = val
            a.lazy[sel] = val
            np.testing.assert_array_equal(a.lazy[sel][...], ref[sel])
        np.testing.assert_array_equal(a[...], ref)

    @pytest.mark.parametrize("cfg", RANDOM_CASES)
    def test_vindex_writes_track_numpy(self, cfg: Config) -> None:
        """Random coordinate (vindex) writes/reads match a NumPy model across boundaries."""
        rng = np.random.default_rng(1)
        a, ref = _make(cfg)
        ref = ref.copy()
        for _ in range(25):
            idx = _unique_coords(rng, cfg.shape, int(rng.integers(1, 7)))
            val = rng.integers(-9999, 9999, size=idx[0].shape, dtype="i4")
            ref[idx] = val
            a.lazy.vindex[idx] = val
            np.testing.assert_array_equal(a.lazy.vindex[idx][...], ref[idx])
        np.testing.assert_array_equal(a[...], ref)

    @pytest.mark.parametrize("cfg", RANDOM_CASES)
    def test_oindex_single_axis_writes_track_numpy(self, cfg: Config) -> None:
        """Random single-fancy-axis orthogonal writes/reads match a NumPy model."""
        rng = np.random.default_rng(2)
        a, ref = _make(cfg)
        ref = ref.copy()
        size0 = cfg.shape[0]
        for _ in range(25):
            k = int(rng.integers(1, size0 + 1))
            idx0 = rng.choice(size0, size=k, replace=False).astype(np.intp)
            sel: Any = (idx0, *(_rand_slice(rng, s) for s in cfg.shape[1:]))
            val = rng.integers(-9999, 9999, size=ref[sel].shape, dtype="i4")
            ref[sel] = val
            a.lazy.oindex[sel] = val
            np.testing.assert_array_equal(a.lazy.oindex[sel][...], ref[sel])
        np.testing.assert_array_equal(a[...], ref)
