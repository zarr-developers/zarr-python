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
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.errors import BoundsCheckError, LazyViewError
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Callable


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
    def test_empty_writes_are_noops(self) -> None:
        a, ref = _make(CONFIGS[1])
        view = a.lazy[2:10]

        view.lazy.vindex[(np.array([], dtype=np.intp),)] = np.array([], dtype="i4")
        view.set_mask_selection(np.zeros(view.shape, dtype=bool), np.array([], dtype="i4"))

        np.testing.assert_array_equal(a[...], ref)

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
        """Composing two lazy slices equals applying them in sequence on NumPy.

        Positive bounds are literal domain coordinates, so the inner slice
        re-selects in the outer view's preserved coordinate system.
        """
        a, ref = _make(cfg)
        n = cfg.shape[0]
        view = a.lazy[1 : n - 1].lazy[2 : n - 2]
        expected = ref[2 : n - 2]  # literal coordinates: the inner slice re-selects
        assert tuple(view.shape) == expected.shape
        np.testing.assert_array_equal(view[...], expected)
        # the composed view's domain is the literal interval it covers
        t = view._async_array._transform
        assert (t.domain.inclusive_min[0], t.domain.exclusive_max[0]) == (2, n - 2)


class TestLazyViewMethods:
    """Indexing methods called on a lazy *view* object (``v = arr.lazy[...]``) must
    honor the view's transform, not silently fall back to the storage grid.

    The accessor (``arr.lazy.oindex[...]``) was covered elsewhere, but the methods
    on the *returned* non-identity Array (``v.oindex[...]``, ``v[..., -1]``) route
    through ``Array``'s own dispatch, which had correctness gaps.
    """

    def test_zero_rank_result_is_scalar(self) -> None:
        """A zero-dimensional array reads back as a scalar (shape ``()``) eagerly and lazily."""
        a = zarr.create_array({}, shape=(), chunks=(), dtype="i4")
        a[...] = 5
        assert np.shape(a[...]) == ()
        assert np.shape(a[()]) == ()
        assert np.shape(a.lazy[...].result()) == ()
        assert a.lazy[...].result() == 5

    def test_coordinate_methods_wrap_negative_indices(self) -> None:
        a, ref = _make(CONFIGS[1])
        view = a.lazy[2:10]

        np.testing.assert_array_equal(
            view.get_coordinate_selection((np.array([-1], dtype=np.intp),)), ref[[9]]
        )
        view.set_coordinate_selection((np.array([-1], dtype=np.intp),), 999)

        expected = ref.copy()
        expected[9] = 999
        np.testing.assert_array_equal(a[...], expected)

    @pytest.mark.parametrize("cfg", ND_CASES)
    def test_view_oindex_respects_transform(self, cfg: Config) -> None:
        """``v.oindex[idx]`` on a sub-view reads from the view, not the base array."""
        a, ref = _make(cfg)
        n0 = cfg.shape[0]
        cut = n0 // 2
        vslice = (slice(cut, n0), *([slice(None)] * (len(cfg.shape) - 1)))
        v = a.lazy[vslice]
        idx = np.array([cut, cfg.shape[0] - 1], dtype=np.intp)  # domain coordinates
        osel: Any = (idx, *([slice(None)] * (len(cfg.shape) - 1)))
        np.testing.assert_array_equal(v.oindex[osel], ref[osel])

    @pytest.mark.parametrize("cfg", MULTI_AXIS_CASES)
    def test_view_trailing_index_after_ellipsis(self, cfg: Config) -> None:
        """``v[..., k]``: positive ``k`` is a literal domain coordinate; a negative
        ``k`` wraps from the end of the view's domain (NumPy parity at the boundary)."""
        a, ref = _make(cfg)
        v = a.lazy[1 : cfg.shape[0]]
        last = cfg.shape[-1] - 1
        np.testing.assert_array_equal(v[..., last], ref[1:, ..., last])
        np.testing.assert_array_equal(v[..., -1], ref[1:, ..., -1])

    @pytest.mark.parametrize("cfg", MULTI_AXIS_CASES)
    def test_view_basic_tuple_and_int(self, cfg: Config) -> None:
        """Basic tuple selections on a view (incl. integer axes that drop) honor the view."""
        a, ref = _make(cfg)
        cut = cfg.shape[0] // 2
        v = a.lazy[cut:]
        full: Any = (slice(cut, cut + 2), *([slice(None)] * (len(cfg.shape) - 1)))
        np.testing.assert_array_equal(v[full], ref[full])
        intsel: Any = (cut + 1, *([slice(None)] * (len(cfg.shape) - 1)))
        np.testing.assert_array_equal(v[intsel], ref[intsel])  # int axis drops

    @pytest.mark.parametrize("cfg", MULTI_AXIS_CASES)
    def test_view_vindex(self, cfg: Config) -> None:
        """``v.vindex[...]`` (coordinate selection) on a view honors the view."""
        a, ref = _make(cfg)
        cut = cfg.shape[0] // 2
        v = a.lazy[cut:]
        # coordinates: axis 0 lives in [cut, n); the other axes are untouched
        idx = (
            np.array([cut, cut + 1, cut + 2], dtype=np.intp),
            *(np.array([0, 1, 2], dtype=np.intp) for _ in cfg.shape[1:]),
        )
        np.testing.assert_array_equal(v.vindex[idx], ref[idx])

    def test_view_vindex_with_broadcast_out_buffer(self) -> None:
        """vindex with a multi-dim result and out= on a view takes a broadcast-shaped buffer.

        Vectorized indexing scatters through a single flat index internally, but
        the caller-visible `out` contract is the broadcast selection shape: the
        flat temporary is an implementation detail. The out buffer must therefore
        match `expected.shape`, and the results must land in it.
        """
        a, ref = _make(CONFIGS[3])  # 2d-unsharded
        v = a.lazy[2:18]
        i0 = np.array([[2, 3], [4, 5]], dtype=np.intp)  # coordinates within [2, 18)
        i1 = np.array([[0, 5], [10, 15]], dtype=np.intp)
        expected = ref[i0, i1]
        assert expected.shape == (2, 2)
        buf = default_buffer_prototype().nd_buffer.empty(shape=expected.shape, dtype=np.dtype("i4"))
        v.get_coordinate_selection((i0, i1), out=buf)
        np.testing.assert_array_equal(np.asarray(buf.as_ndarray_like()), expected)

    @pytest.mark.parametrize("cfg", MULTI_AXIS_UNSHARDED_CASES)
    def test_view_write_through_tuple(self, cfg: Config) -> None:
        """Writing through a view with a basic tuple selection lands in view coords."""
        a, ref = _make(cfg)
        cut = cfg.shape[0] // 2
        v = a.lazy[cut:]
        expected = ref.copy()
        wsel: Any = (slice(cut, cut + 2), *([slice(None)] * (len(cfg.shape) - 1)))
        val = _value_like(ref[wsel])
        expected[wsel] = val
        v[wsel] = val
        np.testing.assert_array_equal(a[...], expected)


class TestLazyAccessorSurface:
    def test_shape(self) -> None:
        """`.lazy.shape` mirrors the wrapped array's shape, including on views."""
        a, _ = _make(CONFIGS[3])  # 2d-unsharded (20, 30)
        assert a.lazy.shape == (20, 30)
        v = a.lazy[2:8, 5:15]
        assert v.lazy.shape == (6, 10)


class TestDaskInterop:
    def test_from_array_lazy_view(self) -> None:
        """A zero-origin lazy view works as a `dask.array.from_array` source.

        This is the dask-free-wrapper use case (napari): the view exposes
        `shape`/`dtype`/`ndim` and eager `__getitem__`, which is all dask needs.
        """
        da = pytest.importorskip("dask.array")
        a, ref = _make(CONFIGS[3])  # 2d-unsharded (20, 30)
        v = a.lazy[2:18, 5:25].translate_to((0, 0))
        d = da.from_array(v, chunks=(8, 10))
        assert d.shape == (16, 20)
        np.testing.assert_array_equal(d.compute(scheduler="synchronous"), ref[2:18, 5:25])


class TestLazyErrors:
    def test_negative_step_raises(self) -> None:
        """A negative slice step is unsupported and raises on the lazy path."""
        a, _ = _make(CONFIGS[1])  # 1d-unsharded
        with pytest.raises(IndexError, match="step must be positive"):
            a.lazy[::-1]

    def test_accessor_negative_index_wraps(self) -> None:
        """The public boundary wraps negative indices NumPy-style against the
        current view's ``exclusive_max`` (positive indices stay literal domain
        coordinates). Only out-of-range negatives (``k < -size``) or positive
        indices past the end raise — before any chunk access.
        """
        a, ref = _make(CONFIGS[1])  # 1d-unsharded, shape (24,)
        np.testing.assert_array_equal(a.lazy[-1].result(), ref[-1])
        np.testing.assert_array_equal(
            a.lazy.oindex[(np.array([-1], dtype=np.intp),)].result(), ref[[-1]]
        )
        np.testing.assert_array_equal(
            a.lazy.vindex[(np.array([-1], dtype=np.intp),)].result(), ref[[-1]]
        )
        # too-negative (< -size) and positive-past-the-end still raise
        for sel in (np.array([-25], dtype=np.intp), np.array([24], dtype=np.intp)):
            with pytest.raises(IndexError, match="out of bounds"):
                _ = a.lazy.oindex[(sel,)].result()
            with pytest.raises(IndexError, match="out of bounds"):
                _ = a.lazy.vindex[(sel,)].result()

    def test_negative_slice_bounds_wrap(self) -> None:
        """Negative slice bounds wrap from the end of the current view's domain
        (NumPy parity at the boundary). Bounds that wrap out of range still raise,
        and bounds that resolve reversed are invalid intervals.
        """
        a, ref = _make(CONFIGS[1])  # 1d-unsharded, shape (24,)
        for sel in (slice(-3, None), slice(-24, None), slice(None, -1), slice(1, -1)):
            np.testing.assert_array_equal(a.lazy[sel].result(), ref[sel])
            np.testing.assert_array_equal(a.lazy.oindex[(sel,)].result(), ref[sel])
        # on a view: wraps against the view's exclusive_max, not the base's
        v = a.lazy[2:10]  # domain [2, 10)
        np.testing.assert_array_equal(v.lazy[-2:].result(), ref[8:10])
        # writes wrap too
        a.lazy[-3:] = 0
        expected = ref.copy()
        expected[-3:] = 0
        np.testing.assert_array_equal(a[...], expected)
        # out-of-range wrap (start < -size) still raises
        with pytest.raises(BoundsCheckError, match="not contained"):
            a.lazy[-25:]
        # bounds that RESOLVE reversed (stop < start after wrapping) are invalid
        with pytest.raises(IndexError, match="interval"):
            a.lazy[-1:1]

    def test_slice_bounds_strict_containment(self) -> None:
        """Non-empty slice intervals must be contained in the domain — no
        clamping (TensorStore semantics); empty intervals are valid anywhere;
        reversed bounds are an error, not an empty result."""
        a, _ = _make(CONFIGS[1])  # shape (24,)
        for sel in (slice(5, 100), slice(100, 200), slice(0, 25)):
            with pytest.raises(BoundsCheckError, match="not contained"):
                a.lazy[sel]
        assert a.lazy[5:5].shape == (0,)
        assert a.lazy[30:30].shape == (0,)  # empty is valid even outside the domain
        with pytest.raises(IndexError, match="interval"):
            a.lazy[5:2]

    def test_view_positive_literal_negative_wraps(self) -> None:
        """A view's positive indices are literal domain coordinates; negative
        indices wrap from the view's ``exclusive_max`` (NumPy parity at the
        boundary). Positive coordinates below the domain origin remain out of
        bounds — the view keeps its preserved coordinate system for positives.
        """
        a, ref = _make(CONFIGS[1])
        v = a.lazy[2:10]  # domain [2, 10)
        np.testing.assert_array_equal(v[2:5], ref[2:5])  # positive literal
        np.testing.assert_array_equal(v[3], ref[3])  # positive literal
        np.testing.assert_array_equal(v[-1], ref[9])  # wrap: exclusive_max 10 - 1
        np.testing.assert_array_equal(v[-3:], ref[7:10])  # wrap
        # positive coordinates below the origin are still out of bounds
        below_origin: list[Callable[[], Any]] = [lambda: v[1], lambda: v[0:3]]
        for bad in below_origin:
            with pytest.raises(BoundsCheckError):
                bad()
        np.testing.assert_array_equal(np.asarray(v), ref[2:10])
        z = v.translate_to((0,))
        np.testing.assert_array_equal(z[0:3], ref[2:5])

    def test_lazy_bounds_errors_share_one_type(self) -> None:
        """Out-of-range indices — positive past the end, or negative past
        ``-size`` — all raise BoundsCheckError (an IndexError), with one message
        shape naming the valid range."""
        a, _ = _make(CONFIGS[1])  # shape (24,)
        triggers: list[Callable[[], Any]] = [
            lambda: a.lazy[24],
            lambda: a.lazy[-25],
            lambda: a.lazy[20:30],
            lambda: a.lazy.oindex[(np.array([-25], dtype=np.intp),)],
        ]
        for trigger in triggers:
            with pytest.raises(BoundsCheckError, match="out of bounds|not contained"):
                trigger()

    def test_guard_message_names_real_apis(self) -> None:
        """LazyViewError points at APIs that exist (chunk_projections/metadata),
        not the not-yet-added chunk_layout."""
        a, _ = _make(CONFIGS[1])
        with pytest.raises(LazyViewError) as ei:
            _ = a.lazy[2:10].chunks
        assert "chunk_projections" in str(ei.value)
        assert "chunk_layout" not in str(ei.value)

    def test_mask_positions_are_absolute_coordinates(self) -> None:
        """Boolean-mask True-positions are absolute coordinates counted from 0,
        NOT offsets from the view's origin (TensorStore semantics, verified
        against tensorstore 0.1.84: on a domain-[2,10) view, a mask with True
        at {3,5,7} addresses cells 3,5,7 — and a True below the origin is out
        of the domain, rejected eagerly here where TensorStore defers to read).
        """
        a, ref = _make(CONFIGS[1])  # shape (24,)
        v = a.lazy[2:10]  # domain [2, 10)
        mask = np.zeros(8, dtype=bool)
        mask[[3, 5, 7]] = True
        np.testing.assert_array_equal(v.lazy.oindex[mask].result(), ref[[3, 5, 7]])
        v.lazy.oindex[mask] = 99  # write path: same coordinates
        assert list(np.flatnonzero(a[...] == 99)) == [3, 5, 7]
        below_origin = np.zeros(8, dtype=bool)
        below_origin[0] = True  # coordinate 0 is not in [2, 10)
        with pytest.raises(BoundsCheckError, match="out of bounds"):
            v.lazy.oindex[below_origin]

    def test_views_are_not_iterable(self) -> None:
        """iter() on a view raises eagerly (TensorStore parity): the getitem
        protocol counts positions from 0, which are not domain coordinates."""
        a, ref = _make(CONFIGS[1])
        with pytest.raises(TypeError, match="not iterable"):
            iter(a.lazy[2:10])
        assert [int(x) for x in a][:3] == [int(v) for v in ref[:3]]  # base unchanged

    def test_fancy_view_repr_does_not_crash(self) -> None:
        """repr of an integer-indexed fancy view must not raise (0-d index array
        in selection_repr)."""
        b = zarr.create_array({}, shape=(6, 8), chunks=(2, 3), dtype="i4")
        b[...] = np.arange(48, dtype="i4").reshape(6, 8)
        ov = b.lazy.oindex[np.array([1, 3]), slice(None)]
        assert isinstance(repr(ov.lazy[0]), str)


class TestLazyComposedAdvanced:
    """Basic-slice-then-fancy composition routes through the view's transform
    (the view's domain origin is preserved), with negatives wrapped at the
    boundary against the view's ``exclusive_max``."""

    def test_slice_then_oindex_read(self) -> None:
        """``v.oindex[idx]`` on a sliced view reads in the view's domain
        coordinates; negatives wrap from the view's end."""
        a, ref = _make(CONFIGS[1])  # shape (24,)
        v = a.lazy[10:20]  # domain [10, 20)
        np.testing.assert_array_equal(v.oindex[np.array([10, 15, 19])], ref[[10, 15, 19]])
        np.testing.assert_array_equal(v.oindex[np.array([-1, -10])], ref[[19, 10]])

    def test_slice_then_lazy_oindex_read(self) -> None:
        """The lazy accessor chain ``v.lazy.oindex[idx]`` composes the same way."""
        a, ref = _make(CONFIGS[1])
        v = a.lazy[10:20]
        np.testing.assert_array_equal(v.lazy.oindex[np.array([-1, -10])].result(), ref[[19, 10]])

    def test_slice_then_oindex_write(self) -> None:
        """Writes through the composed view land at the wrapped domain coordinates."""
        a, ref = _make(CONFIGS[1])
        v = a.lazy[10:20]
        v.oindex[np.array([-1, -10])] = np.array([777, 888], dtype="i4")
        expected = ref.copy()
        expected[19] = 777
        expected[10] = 888
        np.testing.assert_array_equal(a[...], expected)

    def test_slice_then_oindex_out_of_bounds(self) -> None:
        """Coordinates outside the view's ``[10, 20)`` domain raise before I/O:
        a positive below the origin, a negative wrapping below the origin, and a
        positive past the end."""
        a, _ = _make(CONFIGS[1])
        v = a.lazy[10:20]
        for sel in (np.array([0]), np.array([-11]), np.array([20])):
            with pytest.raises(BoundsCheckError):
                _ = v.oindex[sel]


class TestLazyBoolScalar:
    """`isinstance(True, int)` is True, so a bare boolean scalar would otherwise
    pass the integer validators and silently read element 0/1. Reject it at the
    boundary; boolean *arrays* (masks) stay valid."""

    @pytest.mark.parametrize("accessor", ["basic", "oindex", "vindex"])
    @pytest.mark.parametrize("val", [True, False, np.True_, np.False_])
    def test_bool_scalar_rejected(self, accessor: str, val: Any) -> None:
        a, _ = _make(CONFIGS[1])
        acc: Any = {"basic": a.lazy, "oindex": a.lazy.oindex, "vindex": a.lazy.vindex}[accessor]
        with pytest.raises(IndexError):
            _ = acc[val]

    def test_bool_array_still_valid(self) -> None:
        """A boolean *array* is a mask, not a scalar, and remains a valid index."""
        a, ref = _make(CONFIGS[1])
        mask = np.zeros(24, dtype=bool)
        mask[[3, 5, 7]] = True
        np.testing.assert_array_equal(a.lazy.oindex[mask].result(), ref[[3, 5, 7]])


class TestLazyMaskShape:
    """A boolean mask must exactly match the view domain's shape on the
    dimension(s) it consumes (NumPy parity: "boolean index did not match indexed
    array"). Anything else raises IndexError at the boundary, before transform
    construction — never a silent truncation."""

    def test_correct_shape_masks_work(self) -> None:
        """Exact-shape masks select as before, through oindex, vindex, and on a
        non-zero-origin view (whose domain shape, not the base's, is the ruler)."""
        a, ref = _make(CONFIGS[1])  # shape (24,)
        mask = np.zeros(24, dtype=bool)
        mask[[3, 5, 7]] = True
        np.testing.assert_array_equal(a.lazy.oindex[mask].result(), ref[[3, 5, 7]])
        np.testing.assert_array_equal(a.lazy.vindex[mask].result(), ref[[3, 5, 7]])
        v = a.lazy[2:10]  # domain [2, 10) -> shape (8,)
        vmask = np.zeros(8, dtype=bool)
        vmask[[3, 5]] = True  # True positions are absolute coordinates
        np.testing.assert_array_equal(v.lazy.oindex[vmask].result(), ref[[3, 5]])

    @pytest.mark.parametrize("accessor", ["oindex", "vindex"])
    def test_under_length_mask_raises(self, accessor: str) -> None:
        a, _ = _make(CONFIGS[1])  # shape (24,)
        acc: Any = getattr(a.lazy, accessor)
        with pytest.raises(IndexError, match="boolean index did not match"):
            _ = acc[np.zeros(5, dtype=bool)]

    @pytest.mark.parametrize("accessor", ["oindex", "vindex"])
    @pytest.mark.parametrize("fill", [False, True])
    def test_over_length_mask_raises(self, accessor: str, fill: bool) -> None:
        """Over-length masks raise the shape-mismatch error regardless of where
        their True values fall (not a bounds error on the True positions)."""
        a, _ = _make(CONFIGS[1])
        acc: Any = getattr(a.lazy, accessor)
        with pytest.raises(IndexError, match="boolean index did not match"):
            _ = acc[np.full(30, fill, dtype=bool)]

    @pytest.mark.parametrize("accessor", ["oindex", "vindex"])
    def test_wrong_ndim_mask_raises(self, accessor: str) -> None:
        """A 2-D mask on a 1-D array is an IndexError (it consumes two
        dimensions), not a ValueError from deeper in the transform layer."""
        a, _ = _make(CONFIGS[1])
        acc: Any = getattr(a.lazy, accessor)
        with pytest.raises(IndexError, match="too many indices"):
            _ = acc[np.zeros((4, 6), dtype=bool)]

    def test_view_mask_checked_against_view_domain(self) -> None:
        """On a view, the mask ruler is the view's domain shape: the base
        array's shape is the wrong length there."""
        a, _ = _make(CONFIGS[1])
        v = a.lazy[2:10]  # domain shape (8,)
        with pytest.raises(IndexError, match="boolean index did not match"):
            _ = v.lazy.oindex[np.zeros(24, dtype=bool)]


_GET_FIELD_METHODS = [
    "get_basic_selection",
    "get_orthogonal_selection",
    "get_mask_selection",
    "get_coordinate_selection",
]
_SET_FIELD_METHODS = [
    "set_basic_selection",
    "set_orthogonal_selection",
    "set_mask_selection",
    "set_coordinate_selection",
]


class TestLazyFieldsOnView:
    """`fields=` on a non-identity view routed through the legacy indexer, which
    ignores the transform and reads/writes the wrong storage region. It must now
    raise NotImplementedError before any storage access in all eight
    get/set_*_selection methods (and the ``v['f0', :]`` sugar), leaving the base
    array bit-identical."""

    @staticmethod
    def _struct_array() -> tuple[zarr.Array[Any], npt.NDArray[Any]]:
        dt = np.dtype([("f0", "i4"), ("f1", "i4")])
        a = zarr.create_array(MemoryStore(), shape=(10,), chunks=(10,), dtype=dt, zarr_format=2)
        base = np.zeros(10, dtype=dt)
        base["f0"] = np.arange(10, dtype="i4")
        base["f1"] = np.arange(10, dtype="i4") + 100
        a[...] = base
        return a, base

    @staticmethod
    def _selection_for(method: str) -> Any:
        if "mask" in method:
            return np.zeros(5, dtype=bool)  # view shape is (5,)
        if "coordinate" in method:
            return (np.array([5, 6, 7], dtype=np.intp),)
        return Ellipsis

    @pytest.mark.parametrize("method", _GET_FIELD_METHODS)
    def test_get_fields_on_view_raises(self, method: str) -> None:
        a, base = self._struct_array()
        v = a.lazy[5:10]
        with pytest.raises(NotImplementedError, match="field"):
            getattr(v, method)(self._selection_for(method), fields="f0")
        np.testing.assert_array_equal(a[...], base)

    @pytest.mark.parametrize("method", _SET_FIELD_METHODS)
    def test_set_fields_on_view_raises(self, method: str) -> None:
        a, _base = self._struct_array()
        before = np.asarray(a[...]).copy()
        v = a.lazy[5:10]
        with pytest.raises(NotImplementedError, match="field"):
            getattr(v, method)(self._selection_for(method), np.arange(3, dtype="i4"), fields="f0")
        np.testing.assert_array_equal(a[...], before)

    def test_getitem_field_sugar_on_view_raises(self) -> None:
        a, base = self._struct_array()
        v = a.lazy[5:10]
        selections: list[Any] = [("f0", slice(None)), "f0"]
        for sel in selections:
            with pytest.raises(NotImplementedError, match="field"):
                _ = v[sel]
        np.testing.assert_array_equal(a[...], base)

    def test_setitem_field_sugar_on_view_raises(self) -> None:
        a, _base = self._struct_array()
        before = np.asarray(a[...]).copy()
        v = a.lazy[5:10]
        field_sel: Any = ("f0", slice(None))
        with pytest.raises(NotImplementedError, match="field"):
            v[field_sel] = np.arange(5, dtype="i4")
        np.testing.assert_array_equal(a[...], before)

    def test_fields_on_identity_array_still_work(self) -> None:
        """Field selection on the base (identity) array still routes to the legacy
        path — the view guard must not block it. (Only the write path is checked
        here; structured-dtype field *reads* are broken independently of this
        change.)"""
        a, _ = self._struct_array()
        a.set_basic_selection(Ellipsis, np.arange(10, dtype="i4") + 1, fields="f0")
        result: Any = np.asarray(a[...])
        np.testing.assert_array_equal(result["f0"], np.arange(10, dtype="i4") + 1)

    @pytest.mark.xfail(
        reason="the eager fields= write path corrupts sibling fields (pre-existing "
        "upstream bug in the legacy identity path, not introduced or fixed by the "
        "lazy-view guard); strict xfail flips when the eager path is fixed",
        strict=True,
    )
    def test_fields_write_preserves_sibling_fields(self) -> None:
        """A `fields='f0'` write must leave `f1` untouched. Documents the known
        eager-path corruption (mirrors the TestKnownFancyIntBugs pinning style)."""
        a, base = self._struct_array()
        a.set_basic_selection(Ellipsis, np.arange(10, dtype="i4") + 1, fields="f0")
        result: Any = np.asarray(a[...])
        np.testing.assert_array_equal(result["f1"], base["f1"])


class TestKnownFancyIntBugs:
    """Strict-xfail pins for the int-on-fancy-picked-dim defect (see review notes):
    integer indexing a dimension that an oindex/vindex selection created is
    mis-lowered, crashing reads or mis-shaping results. These flip to failures
    when the bug is fixed, forcing the pins to be removed."""

    @pytest.mark.xfail(
        reason="integer indexing on an oindex-picked dimension crashes in the "
        "codec pipeline (ArrayMap not collapsed to ConstantMap)",
        strict=True,
    )
    def test_int_read_on_oindex_view(self) -> None:
        """`rows[0]` on an oindex-created view must equal the picked row."""
        b = zarr.create_array({}, shape=(6, 8), chunks=(2, 3), dtype="i4")
        ref = np.arange(48, dtype="i4").reshape(6, 8)
        b[...] = ref
        rows = b.lazy.oindex[np.array([1, 3]), slice(None)]
        np.testing.assert_array_equal(rows[0], ref[[1, 3]][0])

    def test_int_read_on_vindex_view_is_scalar(self) -> None:
        """`pts[0]` on a vindex-created view must be a scalar, as in NumPy."""
        a = zarr.create_array({}, shape=(12,), chunks=(3,), dtype="i4")
        a[...] = np.arange(12, dtype="i4")
        pts = a.lazy.vindex[np.array([1, 3, 5])]
        assert np.shape(pts[0]) == ()
        assert np.shape(pts.lazy[0].result()) == ()
        assert pts[0] == a[1]

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


_VIEW_GUARDED_PROPERTIES = (
    "chunks",
    "shards",
    "read_chunk_sizes",
    "write_chunk_sizes",
    "cdata_shape",
    "nchunks",
    "nchunks_initialized",
    "info",
)

# method name -> call arguments
_VIEW_GUARDED_METHODS: dict[str, tuple[Any, ...]] = {
    "nbytes_stored": (),
    "info_complete": (),
    "resize": ((24,),),
    "append": (np.zeros((3,), dtype="i4"),),
}


class TestLazyViewGridGuards:
    """Grid-describing members assume the array fills its chunk grid, so on a
    non-identity lazy view they must raise instead of silently describing the
    backing grid (a footgun for consumers that size reads off ``.chunks``)."""

    @staticmethod
    def _array() -> zarr.Array[Any]:
        a = zarr.create_array({}, shape=(12,), chunks=(3,), dtype="i4")
        a[...] = np.arange(12, dtype="i4")
        return a

    @pytest.mark.parametrize("name", _VIEW_GUARDED_PROPERTIES)
    def test_property_raises_on_view(self, name: str) -> None:
        """A grid-describing property works on the backing array but raises LazyViewError on a view."""
        a = self._array()
        getattr(a, name)  # backing (identity) array: no raise
        with pytest.raises(LazyViewError):
            getattr(a.lazy[2:10], name)

    @pytest.mark.parametrize("name", list(_VIEW_GUARDED_METHODS))
    def test_method_raises_on_view(self, name: str) -> None:
        """A grid-describing/mutating method raises LazyViewError on a view."""
        a = self._array()
        with pytest.raises(LazyViewError):
            getattr(a.lazy[2:10], name)(*_VIEW_GUARDED_METHODS[name])

    def test_size_and_nbytes_reflect_the_view(self) -> None:
        """size / nbytes are logical members: they describe the view's extent, not the backing array."""
        a = self._array()  # shape (12,), i4 (4 bytes)
        view = a.lazy[2:10]  # logical shape (8,)
        assert view.shape == (8,)
        assert view.size == 8
        assert view.nbytes == 8 * 4


class TestChunkProjections:
    """`chunk_projections` enumerates the stored chunks an (identity or view) array
    projects onto, giving each one's stored region, the region of this array it maps
    to, and whether it is partially covered."""

    @staticmethod
    def _array(shape: tuple[int, ...], chunks: tuple[int, ...]) -> zarr.Array[Any]:
        a = zarr.create_array({}, shape=shape, chunks=chunks, dtype="i4")
        a[...] = np.arange(int(np.prod(shape)), dtype="i4").reshape(shape)
        return a

    def test_identity_tiles_the_whole_domain(self) -> None:
        """On a full (identity) array, projections tile the domain exactly: one per chunk, none partial."""
        a = self._array((10,), (3,))  # chunks (3,3,3,1)
        projs = list(a.chunk_projections())
        assert len(projs) == 4
        assert all(not p.is_partial for p in projs)
        covered = np.zeros(10, dtype=bool)
        for p in projs:
            arr_sel: Any = p.array_selection
            covered[arr_sel] = True
        assert covered.all()  # complete, and (bool assignment) each cell once

    def test_view_round_trip(self) -> None:
        """Placing each projection's stored-chunk slice at its array_selection reconstructs the view."""
        a = self._array((10,), (3,))
        backing = np.asarray(a[...])
        view = a.lazy[2:9]  # (7,)
        out = np.empty(view.shape, dtype="i4")
        for p in view.chunk_projections():
            offset = p.coord[0] * 3  # regular grid: chunk c starts at c*chunk_size
            stored_chunk = backing[offset : offset + p.shape[0]]
            arr_sel: Any = p.array_selection
            chunk_sel: Any = p.chunk_selection
            out[arr_sel] = stored_chunk[chunk_sel]
        np.testing.assert_array_equal(out, backing[2:9])

    def test_partial_flag_and_alignment(self) -> None:
        """Boundary chunks a view only partially covers are flagged; a chunk-aligned view is not."""
        a = self._array((12,), (4,))  # chunks at [0,4) [4,8) [8,12)
        partial = a.lazy[2:12]  # first chunk partially covered
        assert any(p.is_partial for p in partial.chunk_projections())
        assert not partial.is_chunk_aligned()
        aligned = a.lazy[4:12]  # exactly chunks 1 and 2
        assert all(not p.is_partial for p in aligned.chunk_projections())
        assert aligned.is_chunk_aligned()

    def test_sharded_write_unit(self) -> None:
        """For a sharded array, unit='write' enumerates shards; unit='read' (inner chunks) is deferred."""
        a = zarr.create_array({}, shape=(12,), chunks=(2,), shards=(6,), dtype="i4")
        a[...] = np.arange(12, dtype="i4")
        shards = list(a.chunk_projections(unit="write"))
        assert len(shards) == 2  # two shards of size 6
        assert all(p.shape == (6,) and not p.is_partial for p in shards)
        with pytest.raises(NotImplementedError):
            a.chunk_projections(unit="read")

    def test_invalid_unit_rejected(self) -> None:
        """An unknown granularity is rejected eagerly."""
        a = self._array((6,), (2,))
        with pytest.raises(ValueError, match="unit"):
            a.chunk_projections(unit="bogus")  # type: ignore[arg-type]


class TestLazyArrayMapResolution:
    """Chunk resolution of orthogonal (outer-product) and correlated (vectorized)
    ArrayMaps: intersections and output selectors must preserve the distinct
    semantics of the two flavours, across chunk boundaries, and for the
    length-1 shapes that are degenerate under the shape-derived classifier.
    """

    def _make(self, shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[Any, Any]:
        a = zarr.create_array(store={}, shape=shape, chunks=chunks, dtype="i4")
        data = np.arange(int(np.prod(shape)), dtype="i4").reshape(shape)
        a[...] = data
        return a, data

    def test_oindex_multiple_arrays_outer_product(self) -> None:
        """Orthogonal selection with unequal index lengths equals ``np.ix_`` (read)."""
        a, data = self._make((20, 30), (5, 10))
        rows = np.array([1, 11], dtype=np.intp)
        cols = np.array([2, 12, 22], dtype=np.intp)
        actual = a.lazy.oindex[rows, cols].result()
        np.testing.assert_array_equal(actual, data[np.ix_(rows, cols)])

    def test_oindex_multiple_arrays_outer_product_write(self) -> None:
        """Orthogonal outer-product write spanning several chunks (unsharded)."""
        a, data = self._make((20, 30), (5, 10))
        rows = np.array([1, 11], dtype=np.intp)
        cols = np.array([2, 12, 22], dtype=np.intp)
        expected = data.copy()
        val = (np.arange(6, dtype="i4").reshape(2, 3) + 1) * 100
        expected[np.ix_(rows, cols)] = val
        a.lazy.oindex[rows, cols] = val
        np.testing.assert_array_equal(a[...], expected)

    def test_oindex_length1_and_length3_axes(self) -> None:
        """A length-1 orthogonal axis (all-singleton shape) must not be
        misclassified as scalar/correlated: the outer product still holds."""
        a, data = self._make((6, 6), (2, 2))
        rows = np.array([2], dtype=np.intp)
        cols = np.array([1, 3, 5], dtype=np.intp)
        actual = a.lazy.oindex[rows, cols].result()
        np.testing.assert_array_equal(actual, data[np.ix_(rows, cols)])
        assert actual.shape == (1, 3)

    def test_vindex_length1_pair(self) -> None:
        """Two length-1 correlated arrays (degenerate (1,) shape) stay a single
        pointwise scatter, not an outer product."""
        a, data = self._make((6, 6), (2, 2))
        actual = a.lazy.vindex[np.array([2]), np.array([3])].result()
        np.testing.assert_array_equal(actual, data[np.array([2]), np.array([3])])

    def test_vindex_two_arrays_with_residual_slice(self) -> None:
        """Vectorized selection with two correlated arrays plus a residual slice
        dim (partial indexing of a 3-D array) matches NumPy fancy indexing."""
        a, data = self._make((4, 3, 5), (2, 2, 2))
        rows = np.array([1, 3], dtype=np.intp)
        cols = np.array([2, 0], dtype=np.intp)
        actual = a.lazy.vindex[rows, cols].result()
        np.testing.assert_array_equal(actual, data[rows, cols])
        assert actual.shape == (2, 5)

    def test_vindex_two_arrays_with_residual_slice_write(self) -> None:
        """Write-through of the correlated + residual-slice case."""
        a, data = self._make((4, 3, 5), (2, 2, 2))
        rows = np.array([1, 3], dtype=np.intp)
        cols = np.array([2, 0], dtype=np.intp)
        expected = data.copy()
        val = (np.arange(10, dtype="i4").reshape(2, 5) + 1) * 100
        expected[rows, cols] = val
        a.lazy.vindex[rows, cols] = val
        np.testing.assert_array_equal(a[...], expected)

    def test_vindex_partial_rank_bool_mask(self) -> None:
        """A partial-rank boolean mask (rank < array ndim) leaves a residual slice
        dim; the vectorized read matches NumPy."""
        a, data = self._make((4, 3, 5), (2, 2, 2))
        mask = np.zeros((4, 3), dtype=bool)
        mask[1, 2] = mask[3, 0] = True
        actual = a.lazy.vindex[mask].result()
        np.testing.assert_array_equal(actual, data[mask])
        assert actual.shape == (2, 5)

    def test_basic_slice_after_oindex_independent_axis(self) -> None:
        """Basic-slicing an axis the ArrayMap is independent of broadcasts the
        map (does not slice its values); the slice narrows only that axis."""
        a, data = self._make((6, 6), (2, 2))
        cols = a.lazy.oindex[:, np.array([1, 3, 5], dtype=np.intp)]
        ref = data[:, [1, 3, 5]]
        np.testing.assert_array_equal(cols.lazy[2:4, :].result(), ref[2:4, :])

    def test_basic_slice_after_oindex_fancy_axis(self) -> None:
        """Basic-slicing the fancy axis of an ArrayMap slices the map's values."""
        a, data = self._make((6, 6), (2, 2))
        cols = a.lazy.oindex[:, np.array([1, 3, 5], dtype=np.intp)]
        ref = data[:, [1, 3, 5]]
        np.testing.assert_array_equal(cols.lazy[:, 0:2].result(), ref[:, 0:2])

    def test_empty_oindex_chunk_projections(self) -> None:
        """An empty fancy selection yields no chunk projections (no crash)."""
        a = zarr.create_array(store={}, shape=(10,), chunks=(3,), dtype="i4")
        assert list(a.lazy.oindex[np.array([], dtype=np.intp)].chunk_projections()) == []

    def test_empty_oindex_read(self) -> None:
        """An empty fancy selection reads back an empty array."""
        a, _ = self._make((10,), (3,))
        actual = a.lazy.oindex[np.array([], dtype=np.intp)].result()
        assert actual.shape == (0,)


# ---------------------------------------------------------------------------
# Async-surface guards
# ---------------------------------------------------------------------------
#
# The async surface (``AsyncArray.getitem`` / ``setitem`` / selection methods and
# the ``AsyncOIndex`` / ``AsyncVIndex`` accessors) does not yet route selections
# through a view's index transform: it builds a legacy indexer from
# ``metadata.shape``, which for a lazy view silently reads or writes the *wrong*
# region. Until async transform routing lands, every such entry point must raise
# ``LazyViewError`` for a non-identity view, steering users to the synchronous
# ``Array`` surface (or ``.result()``). Identity arrays must be unaffected.


def _read_ops() -> list[Any]:
    """Async read entry points that build a legacy indexer, keyed by name."""
    coord = (np.array([0, 1], dtype=np.intp),)
    mask = np.zeros((10,), dtype=bool)
    mask[0] = mask[2] = True
    return [
        pytest.param(lambda av: av.getitem(slice(None)), id="getitem"),
        pytest.param(
            lambda av: av.get_orthogonal_selection((slice(0, 3),)), id="get_orthogonal_selection"
        ),
        pytest.param(lambda av: av.get_coordinate_selection(coord), id="get_coordinate_selection"),
        pytest.param(lambda av: av.get_mask_selection(mask), id="get_mask_selection"),
        pytest.param(lambda av: av.oindex.getitem((slice(0, 3),)), id="oindex.getitem"),
        pytest.param(lambda av: av.vindex.getitem(coord), id="vindex.getitem"),
    ]


class TestAsyncSurfaceGuards:
    def _view(self) -> tuple[zarr.Array[Any], npt.NDArray[Any]]:
        """A base array and a non-identity async view ``arr.lazy[2:12]``'s async_array."""
        a = zarr.create_array(MemoryStore(), shape=(24,), chunks=(4,), dtype="i4")
        ref = np.arange(24, dtype="i4")
        a[...] = ref
        return a, ref

    @pytest.mark.parametrize("op", _read_ops())
    def test_read_entry_points_raise(self, op: Callable[[Any], Any]) -> None:
        """Every async read entry point raises ``LazyViewError`` on a lazy view."""
        a, _ = self._view()
        av = a.lazy[2:12]._async_array
        assert not av._is_identity
        with pytest.raises(LazyViewError):
            sync(op(av))

    def test_setitem_raises_and_leaves_base_unchanged(self) -> None:
        """``AsyncArray.setitem`` on a view raises and writes nothing to the base."""
        a, ref = self._view()
        av = a.lazy[5:10]._async_array
        with pytest.raises(LazyViewError):
            sync(av.setitem(slice(0, 5), 99))
        np.testing.assert_array_equal(a[...], ref)

    def test_from_array_view_explicit_kwargs_raises_and_writes_nothing(self) -> None:
        """``from_array`` with a lazy-view source raises and leaves the target empty."""
        a, _ = self._view()
        view = a.lazy[5:10]
        target = MemoryStore()
        with pytest.raises(LazyViewError):
            zarr.from_array(target, data=view, chunks=(5,), shards=None, overwrite=False)
        assert target._store_dict == {}

    def test_from_array_view_default_kwargs_raises_by_design(self) -> None:
        """The default ('keep') kwarg path raises the same clear error, not by accident."""
        a, _ = self._view()
        view = a.lazy[5:10]
        target = MemoryStore()
        with pytest.raises(LazyViewError):
            zarr.from_array(target, data=view)
        assert target._store_dict == {}

    def test_translate_by_async_view_getitem_raises(self) -> None:
        """A view produced by ``AsyncArray.translate_by`` is guarded too."""
        a, _ = self._view()
        av = a._async_array.translate_by((3,))
        assert not av._is_identity
        with pytest.raises(LazyViewError):
            sync(av.getitem(slice(None)))

    # --- identity regression guards: none of the above may affect eager arrays ---

    def test_identity_getitem_still_reads(self) -> None:
        a, ref = self._view()
        got = sync(a._async_array.getitem(slice(2, 12)))
        np.testing.assert_array_equal(got, ref[2:12])

    def test_identity_setitem_still_writes(self) -> None:
        a, ref = self._view()
        sync(a._async_array.setitem(slice(0, 5), 99))
        expected = ref.copy()
        expected[0:5] = 99
        np.testing.assert_array_equal(a[...], expected)

    def test_identity_from_array_still_copies(self) -> None:
        a, ref = self._view()
        target = MemoryStore()
        out = zarr.from_array(target, data=a)
        np.testing.assert_array_equal(out[...], ref)
