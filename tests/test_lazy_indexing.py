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

import numpy as np
import numpy.typing as npt
import pytest

import zarr
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
# Orthogonal indexing across >=2 array axes is currently broken (see
# TestLazyOIndex.test_multi_axis_read_xfail); these are the configs that have
# enough dimensions to exercise it.
MULTI_AXIS_CASES = [pytest.param(cfg, id=cfg.id) for cfg in ND_CONFIGS if len(cfg.shape) >= 2]


def _oindex_one_axis(cfg: Config) -> tuple[Any, ...]:
    """An orthogonal selection with a single fancy axis (axis 0) and slices elsewhere."""
    idx = FANCY_INDICES[len(cfg.shape)][0]
    return (idx, *([slice(None)] * (len(cfg.shape) - 1)))


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

    @pytest.mark.xfail(
        strict=True,
        reason="lazy oindex with >=2 array axes returns a pointwise scatter, not an outer product",
    )
    @pytest.mark.parametrize("cfg", MULTI_AXIS_CASES)
    def test_multi_axis_read_xfail(self, cfg: Config) -> None:
        """Lazy orthogonal indexing across >=2 array axes should equal ``np.ix_``.

        Currently broken: it collapses to the vectorized (pointwise) scatter and
        fills the rest with the fill value. Eager ``oindex`` is correct, so this
        is a lazy-path bug; the strict xfail will flag when it is fixed.
        """
        a, ref = _make(cfg)
        idx = FANCY_INDICES[len(cfg.shape)]
        expected = ref[np.ix_(*idx)]
        view = a.lazy.oindex[idx]
        np.testing.assert_array_equal(view[...], expected)


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


class TestLazyErrors:
    def test_negative_step_raises(self) -> None:
        """A negative slice step is unsupported and raises on the lazy path."""
        a, _ = _make(CONFIGS[1])  # 1d-unsharded
        with pytest.raises(IndexError, match="step must be positive"):
            a.lazy[::-1]
