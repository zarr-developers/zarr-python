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

        Bounds are spelled literally: lazy coordinates are literal, so the
        from-the-end `1:-1` spelling is rejected on the lazy path (see
        TestLazyErrors.test_negative_slice_bounds_are_literal).
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
        """``v[..., k]`` uses literal coordinates; ``-1`` is out of the domain."""
        a, ref = _make(cfg)
        v = a.lazy[1 : cfg.shape[0]]
        last = cfg.shape[-1] - 1
        np.testing.assert_array_equal(v[..., last], ref[1:, ..., last])
        with pytest.raises(BoundsCheckError):
            v[..., -1]

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

    def test_view_vindex_with_flat_out_buffer(self) -> None:
        """vindex with a multi-dim result and out= on a view uses a flat out buffer.

        Vectorized indexing scatters through a single flat index, so (as in the
        eager path) the out buffer must be flat (shape = number of points).
        """
        a, ref = _make(CONFIGS[3])  # 2d-unsharded
        v = a.lazy[2:18]
        i0 = np.array([[2, 3], [4, 5]], dtype=np.intp)  # coordinates within [2, 18)
        i1 = np.array([[0, 5], [10, 15]], dtype=np.intp)
        expected = ref[i0, i1]
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
        wsel: Any = (slice(cut, cut + 2), *([slice(None)] * (len(cfg.shape) - 1)))
        val = _value_like(ref[wsel])
        expected[wsel] = val
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

    def test_negative_slice_bounds_are_literal(self) -> None:
        """Negative slice bounds are literal coordinates on the lazy path, like
        every other index form — never from-the-end — so they raise.

        Consistency rule: a lazy selection is a declaration in literal
        coordinates, and view domains start at 0, so a negative value is never
        in bounds regardless of syntactic form (integer, slice bound, or index
        array).
        """
        a, _ = _make(CONFIGS[1])  # 1d-unsharded, shape (24,)
        for sel in (slice(-3, None), slice(-24, None)):
            with pytest.raises(BoundsCheckError, match="not contained"):
                a.lazy[sel]
            with pytest.raises(BoundsCheckError, match="not contained"):
                a.lazy.oindex[(sel,)]
        # bounds that RESOLVE reversed (stop < start), e.g. [1, -1) or [0, -1),
        # are invalid intervals — TensorStore's a[5:2] case — not empties
        for sel in (slice(None, -1), slice(1, -1)):
            with pytest.raises(IndexError, match="interval"):
                a.lazy[sel]
        # ... on views too
        v = a.lazy[2:10]
        with pytest.raises(BoundsCheckError, match="not contained"):
            v.lazy[-2:]
        # ... and for writes
        with pytest.raises(BoundsCheckError, match="not contained"):
            a.lazy[-3:] = 0

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

    def test_one_dialect_view_methods_use_domain_coordinates(self) -> None:
        """A view has ONE coordinate system: its preserved domain. Eager methods
        (`v[...]`, `v[k]`) use the same literal coordinates as `.lazy` —
        matching TensorStore, where indexing a view of [2, 10) with 0 or -1 is
        out of bounds. Zero-based NumPy-style access is spelled explicitly:
        materialize (`result()` / `np.asarray`) or re-zero with `translate_to`.
        """
        a, ref = _make(CONFIGS[1])
        v = a.lazy[2:10]
        np.testing.assert_array_equal(v[2:5], ref[2:5])
        np.testing.assert_array_equal(v[3], ref[3])
        bad_reads: list[Callable[[], Any]] = [
            lambda: v[-1],
            lambda: v[0:3],
            lambda: v[-3:],
        ]
        for bad in bad_reads:
            with pytest.raises(BoundsCheckError):
                bad()
        np.testing.assert_array_equal(np.asarray(v), ref[2:10])
        z = v.translate_to((0,))
        np.testing.assert_array_equal(z[0:3], ref[2:5])

    def test_lazy_bounds_errors_share_one_type(self) -> None:
        """All three literal-coordinate rejections raise BoundsCheckError (an
        IndexError), with one message shape naming the valid range."""
        a, _ = _make(CONFIGS[1])
        triggers: list[Callable[[], Any]] = [
            lambda: a.lazy[-1],
            lambda: a.lazy[-3:],
            lambda: a.lazy.oindex[(np.array([-1], dtype=np.intp),)],
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

    @pytest.mark.xfail(
        reason="vindex-created views return shape-(1,) data for integer reads "
        "where NumPy returns a scalar",
        strict=True,
    )
    def test_int_read_on_vindex_view_is_scalar(self) -> None:
        """`pts[0]` on a vindex-created view must be a scalar, as in NumPy."""
        a = zarr.create_array({}, shape=(12,), chunks=(3,), dtype="i4")
        a[...] = np.arange(12, dtype="i4")
        pts = a.lazy.vindex[np.array([1, 3, 5])]
        assert np.shape(pts[0]) == ()

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
