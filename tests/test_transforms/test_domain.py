from __future__ import annotations

import pytest

from zarr.core.transforms.domain import IndexDomain


class TestIndexDomainConstruction:
    def test_from_shape(self) -> None:
        d = IndexDomain.from_shape((10, 20))
        assert d.inclusive_min == (0, 0)
        assert d.exclusive_max == (10, 20)
        assert d.ndim == 2
        assert d.origin == (0, 0)
        assert d.shape == (10, 20)

    def test_from_shape_0d(self) -> None:
        d = IndexDomain.from_shape(())
        assert d.ndim == 0
        assert d.shape == ()

    def test_non_zero_origin(self) -> None:
        d = IndexDomain(inclusive_min=(5, 10), exclusive_max=(15, 30))
        assert d.origin == (5, 10)
        assert d.shape == (10, 20)
        assert d.ndim == 2

    def test_validation_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            IndexDomain(inclusive_min=(0,), exclusive_max=(10, 20))

    def test_validation_min_greater_than_max(self) -> None:
        with pytest.raises(ValueError, match="inclusive_min must be <="):
            IndexDomain(inclusive_min=(10,), exclusive_max=(5,))

    def test_empty_domain(self) -> None:
        d = IndexDomain(inclusive_min=(5,), exclusive_max=(5,))
        assert d.shape == (0,)

    def test_labels(self) -> None:
        d = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 20), labels=("x", "y"))
        assert d.labels == ("x", "y")

    def test_labels_none(self) -> None:
        d = IndexDomain.from_shape((10,))
        assert d.labels is None


class TestIndexDomainContains:
    def test_contains_inside(self) -> None:
        d = IndexDomain.from_shape((10, 20))
        assert d.contains((0, 0)) is True
        assert d.contains((9, 19)) is True
        assert d.contains((5, 10)) is True

    def test_contains_outside(self) -> None:
        d = IndexDomain.from_shape((10, 20))
        assert d.contains((10, 0)) is False
        assert d.contains((-1, 0)) is False
        assert d.contains((0, 20)) is False

    def test_contains_non_zero_origin(self) -> None:
        d = IndexDomain(inclusive_min=(5,), exclusive_max=(10,))
        assert d.contains((5,)) is True
        assert d.contains((9,)) is True
        assert d.contains((4,)) is False
        assert d.contains((10,)) is False

    def test_contains_wrong_ndim(self) -> None:
        d = IndexDomain.from_shape((10, 20))
        assert d.contains((5,)) is False

    def test_contains_domain_inside(self) -> None:
        outer = IndexDomain.from_shape((10, 20))
        inner = IndexDomain(inclusive_min=(2, 3), exclusive_max=(8, 15))
        assert outer.contains_domain(inner) is True

    def test_contains_domain_outside(self) -> None:
        outer = IndexDomain.from_shape((10, 20))
        inner = IndexDomain(inclusive_min=(2, 3), exclusive_max=(11, 15))
        assert outer.contains_domain(inner) is False

    def test_contains_domain_wrong_ndim(self) -> None:
        outer = IndexDomain.from_shape((10, 20))
        inner = IndexDomain.from_shape((5,))
        assert outer.contains_domain(inner) is False


class TestIndexDomainIntersect:
    def test_overlapping(self) -> None:
        a = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 10))
        b = IndexDomain(inclusive_min=(5, 5), exclusive_max=(15, 15))
        result = a.intersect(b)
        assert result is not None
        assert result.inclusive_min == (5, 5)
        assert result.exclusive_max == (10, 10)

    def test_disjoint(self) -> None:
        a = IndexDomain(inclusive_min=(0,), exclusive_max=(5,))
        b = IndexDomain(inclusive_min=(10,), exclusive_max=(15,))
        assert a.intersect(b) is None

    def test_touching_boundary(self) -> None:
        a = IndexDomain(inclusive_min=(0,), exclusive_max=(5,))
        b = IndexDomain(inclusive_min=(5,), exclusive_max=(10,))
        assert a.intersect(b) is None

    def test_contained(self) -> None:
        a = IndexDomain.from_shape((20,))
        b = IndexDomain(inclusive_min=(5,), exclusive_max=(10,))
        result = a.intersect(b)
        assert result is not None
        assert result.inclusive_min == (5,)
        assert result.exclusive_max == (10,)

    def test_wrong_ndim(self) -> None:
        a = IndexDomain.from_shape((10,))
        b = IndexDomain.from_shape((10, 20))
        with pytest.raises(ValueError, match="different ranks"):
            a.intersect(b)


class TestIndexDomainTranslate:
    def test_translate_positive(self) -> None:
        d = IndexDomain.from_shape((10, 20))
        result = d.translate((5, 10))
        assert result.inclusive_min == (5, 10)
        assert result.exclusive_max == (15, 30)

    def test_translate_negative(self) -> None:
        d = IndexDomain(inclusive_min=(10, 20), exclusive_max=(30, 40))
        result = d.translate((-10, -20))
        assert result.inclusive_min == (0, 0)
        assert result.exclusive_max == (20, 20)

    def test_translate_wrong_length(self) -> None:
        d = IndexDomain.from_shape((10,))
        with pytest.raises(ValueError, match="same length"):
            d.translate((1, 2))


class TestIndexDomainNarrow:
    def test_narrow_slice(self) -> None:
        d = IndexDomain.from_shape((10, 20))
        result = d.narrow((slice(2, 8), slice(5, 15)))
        assert result.inclusive_min == (2, 5)
        assert result.exclusive_max == (8, 15)

    def test_narrow_int(self) -> None:
        d = IndexDomain.from_shape((10, 20))
        result = d.narrow((3, slice(None)))
        assert result.inclusive_min == (3, 0)
        assert result.exclusive_max == (4, 20)

    def test_narrow_ellipsis(self) -> None:
        d = IndexDomain.from_shape((10, 20, 30))
        result = d.narrow((slice(1, 5), ...))
        assert result.inclusive_min == (1, 0, 0)
        assert result.exclusive_max == (5, 20, 30)

    def test_narrow_slice_none(self) -> None:
        d = IndexDomain.from_shape((10,))
        result = d.narrow((slice(None),))
        assert result == d

    def test_narrow_non_zero_origin(self) -> None:
        d = IndexDomain(inclusive_min=(10,), exclusive_max=(20,))
        result = d.narrow((slice(12, 18),))
        assert result.inclusive_min == (12,)
        assert result.exclusive_max == (18,)

    def test_narrow_int_out_of_bounds(self) -> None:
        d = IndexDomain.from_shape((10,))
        with pytest.raises(IndexError, match="out of bounds"):
            d.narrow((10,))

    def test_narrow_int_below_origin(self) -> None:
        d = IndexDomain(inclusive_min=(5,), exclusive_max=(10,))
        with pytest.raises(IndexError, match="out of bounds"):
            d.narrow((4,))

    def test_narrow_clamps_to_domain(self) -> None:
        d = IndexDomain.from_shape((10,))
        result = d.narrow((slice(-5, 100),))
        assert result.inclusive_min == (0,)
        assert result.exclusive_max == (10,)

    def test_narrow_bare_slice(self) -> None:
        d = IndexDomain.from_shape((10,))
        result = d.narrow(slice(2, 8))
        assert result.inclusive_min == (2,)
        assert result.exclusive_max == (8,)

    def test_narrow_too_many_indices(self) -> None:
        d = IndexDomain.from_shape((10,))
        with pytest.raises(IndexError, match="too many indices"):
            d.narrow((1, 2))

    def test_narrow_step_not_one(self) -> None:
        d = IndexDomain.from_shape((10,))
        with pytest.raises(IndexError, match="step=1"):
            d.narrow((slice(0, 10, 2),))
