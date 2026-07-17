"""TensorStore-parity oracle tests for IndexTransform semantics.

Every case in this module was executed against tensorstore 0.1.84 (see the
lazy-indexing design notes): the expected domains, values, and error conditions
are TensorStore's observed behavior, which zarr's lazy indexing matches by
design. Core rules pinned here:

- **Domain preservation**: a step-1 slice keeps the literal coordinates of the
  selected interval (`a[2:10]` has domain `[2, 10)`); nothing re-zeros
  implicitly. Re-zeroing is explicit via `translate_to`.
- **Strided-domain rule**: for step ``k``, ``origin = trunc(start/k)`` (rounded
  toward zero), ``shape = ceil((stop - start)/k)``, and coordinate
  ``origin + i`` maps to base cell ``start + i*k``.
- **Strict containment**: non-empty slice intervals must lie within the domain
  — no clamping, no negative-wrapping; empty intervals are valid anywhere;
  reversed non-empty bounds are an error, not an empty result.
- **Fancy-dim rule**: index-array dims get fresh explicit ``[0, n)`` domains;
  index-array values are absolute domain coordinates.
- **Translate rules**: ``translate_by``/``translate_to`` shift the input domain
  while preserving which cells are addressed.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from zarr_transforms.domain import IndexDomain
from zarr_transforms.errors import BoundsCheckError
from zarr_transforms.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_transforms.transform import IndexTransform


def _identity(lo: int, hi: int) -> IndexTransform:
    """Identity transform over the 1-D domain [lo, hi)."""
    return IndexTransform.identity(IndexDomain(inclusive_min=(lo,), exclusive_max=(hi,)))


def _a() -> IndexTransform:
    """The oracle's base fixture: identity over [0, 12)."""
    return _identity(0, 12)


def _w() -> IndexTransform:
    """The oracle's translated fixture: identity over [-10, 2), cell c -> base c + 10."""
    return _a().translate_domain_by((-10,))


def _dim(t: IndexTransform) -> DimensionMap:
    m = t.output[0]
    assert isinstance(m, DimensionMap)
    return m


def _base_cells(t: IndexTransform) -> list[int]:
    """The base cells a 1-D single-DimensionMap transform addresses, in order."""
    m = _dim(t)
    lo, hi = t.domain.inclusive_min[0], t.domain.exclusive_max[0]
    return [m.offset + m.stride * c for c in range(lo, hi)]


class TestDomainPreservation:
    """Oracle section 1-2: step-1 slices keep literal coordinates."""

    def test_slice_preserves_domain(self) -> None:
        t = _a()[2:10]
        assert (t.domain.inclusive_min, t.domain.exclusive_max) == ((2,), (10,))
        assert _base_cells(t) == list(range(2, 10))

    def test_integer_on_preserved_domain_is_a_coordinate(self) -> None:
        v = _a()[2:10]
        assert isinstance(v[3].output[0], ConstantMap)
        assert v[3].output[0].offset == 3  # coordinate 3 = base cell 3
        assert v[2].output[0].offset == 2
        assert v[9].output[0].offset == 9

    @pytest.mark.parametrize("bad", [0, -1, 10])
    def test_out_of_domain_integer_raises(self, bad: int) -> None:
        with pytest.raises(BoundsCheckError, match=r"valid indices \[2, 10\)"):
            _a()[2:10][bad]

    def test_slice_of_slice_is_literal(self) -> None:
        v = _a()[2:10]
        t = v[3:7]
        assert (t.domain.inclusive_min, t.domain.exclusive_max) == ((3,), (7,))
        assert _base_cells(t) == [3, 4, 5, 6]

    def test_ellipsis_preserves_domain(self) -> None:
        v = _a()[2:10]
        t = v[...]
        assert (t.domain.inclusive_min, t.domain.exclusive_max) == ((2,), (10,))


class TestNegativeOriginDomain:
    """Oracle section 3: on domain [-10, 2), -1 is just another index."""

    def test_translated_domain(self) -> None:
        w = _w()
        assert (w.domain.inclusive_min, w.domain.exclusive_max) == ((-10,), (2,))
        assert _base_cells(w) == list(range(12))

    @pytest.mark.parametrize(("coord", "base"), [(-5, 5), (-10, 0), (-1, 9), (1, 11)])
    def test_negative_coordinates_address_cells(self, coord: int, base: int) -> None:
        t = _w()[coord]
        assert isinstance(t.output[0], ConstantMap)
        assert t.output[0].offset == base

    @pytest.mark.parametrize("bad", [-11, 2])
    def test_out_of_domain_raises(self, bad: int) -> None:
        with pytest.raises(BoundsCheckError, match=r"valid indices \[-10, 2\)"):
            _w()[bad]

    def test_negative_slice_bounds_are_coordinates(self) -> None:
        t = _w()[-5:]
        assert (t.domain.inclusive_min, t.domain.exclusive_max) == ((-5,), (2,))
        assert _base_cells(t) == [5, 6, 7, 8, 9, 10, 11]
        t2 = _w()[-5:-2]
        assert (t2.domain.inclusive_min, t2.domain.exclusive_max) == ((-5,), (-2,))
        assert _base_cells(t2) == [5, 6, 7]


class TestStridedDomains:
    """Oracle section 5: origin = trunc(start/step), coord origin+i -> start + i*step."""

    # (slice, expected (lo, hi), expected base cells) — verbatim oracle rows.
    CASES: ClassVar[list[tuple[slice, tuple[int, int], list[int]]]] = [
        (slice(1, 10, 3), (0, 3), [1, 4, 7]),
        (slice(None, None, 2), (0, 6), [0, 2, 4, 6, 8, 10]),
        (slice(2, 11, 3), (0, 3), [2, 5, 8]),
        (slice(0, 12, 4), (0, 3), [0, 4, 8]),
        (slice(5, 12, 2), (2, 6), [5, 7, 9, 11]),
        (slice(6, 12, 2), (3, 6), [6, 8, 10]),
        (slice(7, 12, 3), (2, 4), [7, 10]),
    ]

    @pytest.mark.parametrize(("sel", "dom", "cells"), CASES)
    def test_strided_domain_and_cells(
        self, sel: slice, dom: tuple[int, int], cells: list[int]
    ) -> None:
        t = _a()[sel]
        assert (t.domain.inclusive_min[0], t.domain.exclusive_max[0]) == dom
        assert _base_cells(t) == cells

    def test_strided_on_negative_origin(self) -> None:
        # w[-9:2:2] -> domain [-4, 2), base cells 1,3,5,7,9,11
        t = _w()[-9:2:2]
        assert (t.domain.inclusive_min[0], t.domain.exclusive_max[0]) == (-4, 2)
        assert _base_cells(t) == [1, 3, 5, 7, 9, 11]
        # w[::2] -> domain [-5, 1), base cells 0,2,4,6,8,10
        t2 = _w()[::2]
        assert (t2.domain.inclusive_min[0], t2.domain.exclusive_max[0]) == (-5, 1)
        assert _base_cells(t2) == [0, 2, 4, 6, 8, 10]

    def test_strided_composition(self) -> None:
        s = _a()[1:10:3]  # domain [0, 3), cells 1,4,7
        assert [s[k].output[0].offset for k in range(3)] == [1, 4, 7]
        t = s[1:3]
        assert (t.domain.inclusive_min[0], t.domain.exclusive_max[0]) == (1, 3)
        assert _base_cells(t) == [4, 7]
        t2 = _a()[::2][1:4]
        assert (t2.domain.inclusive_min[0], t2.domain.exclusive_max[0]) == (1, 4)
        assert _base_cells(t2) == [2, 4, 6]
        t3 = _a()[::2][::2]
        assert (t3.domain.inclusive_min[0], t3.domain.exclusive_max[0]) == (0, 3)
        assert _base_cells(t3) == [0, 4, 8]

    @pytest.mark.parametrize("bad", [-2, -1, 3, 4])
    def test_strided_bounds(self, bad: int) -> None:
        with pytest.raises(BoundsCheckError, match=r"valid indices \[0, 3\)"):
            _a()[1:10:3][bad]


class TestStrictContainment:
    """Oracle section 11: no clamping, no wrapping; empty intervals valid anywhere."""

    @pytest.mark.parametrize(
        "sel",
        [
            slice(5, 100),
            slice(-3, None),
            slice(-3, -1),
            slice(0, 13),
            slice(12, 14),
            slice(100, 200),
        ],
    )
    def test_uncontained_interval_raises(self, sel: slice) -> None:
        with pytest.raises(BoundsCheckError, match="not contained"):
            _a()[sel]

    def test_uncontained_on_negative_origin(self) -> None:
        with pytest.raises(BoundsCheckError, match="not contained"):
            _w()[-20:]

    @pytest.mark.parametrize(
        ("sel", "pos"), [(slice(5, 5), 5), (slice(0, 0), 0), (slice(13, 13), 13)]
    )
    def test_empty_interval_valid_anywhere(self, sel: slice, pos: int) -> None:
        t = _a()[sel]
        assert t.domain.shape == (0,)
        assert t.domain.inclusive_min[0] == pos

    @pytest.mark.parametrize("sel", [slice(5, 2), slice(100, 50)])
    def test_reversed_bounds_raise(self, sel: slice) -> None:
        with pytest.raises(IndexError, match="valid.*interval|interval"):
            _a()[sel]


class TestTranslate:
    """Oracle sections 4 and 12: translate_by / translate_to preserve the cell mapping."""

    def test_translate_to_zero(self) -> None:
        t = _a()[2:10].translate_domain_to((0,))
        assert (t.domain.inclusive_min, t.domain.exclusive_max) == ((0,), (8,))
        assert _base_cells(t) == list(range(2, 10))

    def test_translate_to_offset(self) -> None:
        t = _a().translate_domain_to((5,))
        assert (t.domain.inclusive_min, t.domain.exclusive_max) == ((5,), (17,))
        assert _base_cells(t) == list(range(12))

    def test_translate_by_composes_with_stride(self) -> None:
        # a[::2].translate_by[5] -> domain [5, 11), base = 2*(coord-5)
        t = _a()[::2].translate_domain_by((5,))
        assert (t.domain.inclusive_min, t.domain.exclusive_max) == ((5,), (11,))
        assert _base_cells(t) == [0, 2, 4, 6, 8, 10]
        assert t[5].output[0].offset == 0
        assert t[10].output[0].offset == 10
        with pytest.raises(BoundsCheckError, match=r"valid indices \[5, 11\)"):
            t[0]

    def test_translate_strided_to(self) -> None:
        t = _a()[1:10:3].translate_domain_to((100,))
        assert (t.domain.inclusive_min, t.domain.exclusive_max) == ((100,), (103,))
        assert _base_cells(t) == [1, 4, 7]


class TestFancyDims:
    """Oracle section 7: fancy dims get fresh [0, n); values are absolute coordinates."""

    def test_index_array_values_are_coordinates(self) -> None:
        v = _a()[2:10]
        t = v.oindex[(np.array([3, 5], dtype=np.intp),)]
        assert (t.domain.inclusive_min, t.domain.exclusive_max) == ((0,), (2,))
        m = t.output[0]
        assert isinstance(m, ArrayMap)
        storage = m.offset + m.stride * m.index_array
        np.testing.assert_array_equal(np.asarray(storage).ravel(), [3, 5])

    def test_index_array_on_negative_origin(self) -> None:
        t = _w().oindex[(np.array([-10, -1], dtype=np.intp),)]
        m = t.output[0]
        assert isinstance(m, ArrayMap)
        storage = m.offset + m.stride * m.index_array
        np.testing.assert_array_equal(np.asarray(storage).ravel(), [0, 9])

    def test_index_array_out_of_domain_raises(self) -> None:
        v = _a()[2:10]
        for bad in ([0, 3], [-1, 3], [3, 10]):
            with pytest.raises(BoundsCheckError):
                v.oindex[(np.array(bad, dtype=np.intp),)]
