from __future__ import annotations

from typing import Any

import pytest

from tests.test_transforms.conftest import Expect, ExpectErr
from zarr.core._transforms.domain import IndexDomain, _normalize_selection


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input={"inclusive_min": (0, 0), "exclusive_max": (10, 20)},
            expected={"ndim": 2, "origin": (0, 0), "shape": (10, 20), "labels": None},
            id="2d-zero-origin",
        ),
        Expect(
            input={"inclusive_min": (5, 10), "exclusive_max": (15, 30)},
            expected={"ndim": 2, "origin": (5, 10), "shape": (10, 20), "labels": None},
            id="2d-non-zero-origin",
        ),
        Expect(
            input={"inclusive_min": (5,), "exclusive_max": (5,)},
            expected={"ndim": 1, "origin": (5,), "shape": (0,), "labels": None},
            id="1d-empty",
        ),
        Expect(
            input={"inclusive_min": (), "exclusive_max": ()},
            expected={"ndim": 0, "origin": (), "shape": (), "labels": None},
            id="0d",
        ),
        Expect(
            input={"inclusive_min": (0, 0), "exclusive_max": (10, 20), "labels": ("x", "y")},
            expected={"ndim": 2, "origin": (0, 0), "shape": (10, 20), "labels": ("x", "y")},
            id="2d-with-labels",
        ),
    ],
    ids=lambda c: c.id,
)
def test_construction_success(case: Expect[dict[str, Any], dict[str, Any]]) -> None:
    """IndexDomain construction yields the expected shape, origin, ndim, and labels."""
    d = IndexDomain(**case.input)
    for prop, expected in case.expected.items():
        assert getattr(d, prop) == expected


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input={"inclusive_min": (0,), "exclusive_max": (10, 20)},
            msg="same length",
            exception_cls=ValueError,
            id="mismatched-min-max-lengths",
        ),
        ExpectErr(
            input={"inclusive_min": (10,), "exclusive_max": (5,)},
            msg="inclusive_min must be <=",
            exception_cls=ValueError,
            id="min-greater-than-max",
        ),
        ExpectErr(
            input={"inclusive_min": (0, 0), "exclusive_max": (10, 20), "labels": ("x",)},
            msg="labels must have the same length as dimensions",
            exception_cls=ValueError,
            id="labels-wrong-length",
        ),
    ],
    ids=lambda c: c.id,
)
def test_construction_errors(case: ExpectErr[dict[str, Any]]) -> None:
    """IndexDomain construction with invalid inputs raises ValueError."""
    with pytest.raises(case.exception_cls, match=case.msg):
        IndexDomain(**case.input)


@pytest.mark.parametrize(
    "case",
    [
        Expect(input=(10, 20), expected=(2, (0, 0), (10, 20)), id="2d"),
        Expect(input=(10,), expected=(1, (0,), (10,)), id="1d"),
        Expect(input=(), expected=(0, (), ()), id="0d"),
    ],
    ids=lambda c: c.id,
)
def test_from_shape_success(
    case: Expect[tuple[int, ...], tuple[int, tuple[int, ...], tuple[int, ...]]],
) -> None:
    """IndexDomain.from_shape produces a zero-origin domain with the requested shape."""
    d = IndexDomain.from_shape(case.input)
    expected_ndim, expected_origin, expected_shape = case.expected
    assert d.ndim == expected_ndim
    assert d.origin == expected_origin
    assert d.shape == expected_shape


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(IndexDomain.from_shape((10, 20)), (0, 0)),
            expected=True,
            id="2d-corner-low",
        ),
        Expect(
            input=(IndexDomain.from_shape((10, 20)), (9, 19)),
            expected=True,
            id="2d-corner-high",
        ),
        Expect(
            input=(IndexDomain.from_shape((10, 20)), (5, 10)),
            expected=True,
            id="2d-interior",
        ),
        Expect(
            input=(IndexDomain.from_shape((10, 20)), (10, 0)),
            expected=False,
            id="2d-outside-high",
        ),
        Expect(
            input=(IndexDomain.from_shape((10, 20)), (-1, 0)),
            expected=False,
            id="2d-outside-low",
        ),
        Expect(
            input=(IndexDomain.from_shape((10, 20)), (5,)),
            expected=False,
            id="wrong-ndim",
        ),
        Expect(
            input=(IndexDomain(inclusive_min=(5,), exclusive_max=(10,)), (5,)),
            expected=True,
            id="non-zero-origin-low",
        ),
        Expect(
            input=(IndexDomain(inclusive_min=(5,), exclusive_max=(10,)), (4,)),
            expected=False,
            id="non-zero-origin-below",
        ),
    ],
    ids=lambda c: c.id,
)
def test_contains_success(case: Expect[tuple[IndexDomain, tuple[int, ...]], bool]) -> None:
    """IndexDomain.contains returns True iff the index is within the domain."""
    domain, index = case.input
    assert domain.contains(index) is case.expected


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(
                IndexDomain.from_shape((10, 20)),
                IndexDomain(inclusive_min=(2, 3), exclusive_max=(8, 15)),
            ),
            expected=True,
            id="strict-subset",
        ),
        Expect(
            input=(IndexDomain.from_shape((10, 20)), IndexDomain.from_shape((10, 20))),
            expected=True,
            id="equal-domains",
        ),
        Expect(
            input=(
                IndexDomain.from_shape((10, 20)),
                IndexDomain(inclusive_min=(2, 3), exclusive_max=(11, 15)),
            ),
            expected=False,
            id="extends-past-max",
        ),
        Expect(
            input=(IndexDomain.from_shape((10, 20)), IndexDomain.from_shape((5,))),
            expected=False,
            id="wrong-ndim",
        ),
    ],
    ids=lambda c: c.id,
)
def test_contains_domain_success(case: Expect[tuple[IndexDomain, IndexDomain], bool]) -> None:
    """IndexDomain.contains_domain returns True iff `other` is fully contained."""
    outer, inner = case.input
    assert outer.contains_domain(inner) is case.expected


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(
                IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 10)),
                IndexDomain(inclusive_min=(5, 5), exclusive_max=(15, 15)),
            ),
            expected=IndexDomain(inclusive_min=(5, 5), exclusive_max=(10, 10)),
            id="overlapping-2d",
        ),
        Expect(
            input=(
                IndexDomain.from_shape((20,)),
                IndexDomain(inclusive_min=(5,), exclusive_max=(10,)),
            ),
            expected=IndexDomain(inclusive_min=(5,), exclusive_max=(10,)),
            id="contained",
        ),
        Expect(
            input=(
                IndexDomain(inclusive_min=(0,), exclusive_max=(5,)),
                IndexDomain(inclusive_min=(10,), exclusive_max=(15,)),
            ),
            expected=None,
            id="disjoint",
        ),
        Expect(
            input=(
                IndexDomain(inclusive_min=(0,), exclusive_max=(5,)),
                IndexDomain(inclusive_min=(5,), exclusive_max=(10,)),
            ),
            expected=None,
            id="touching-boundary",
        ),
    ],
    ids=lambda c: c.id,
)
def test_intersect_success(
    case: Expect[tuple[IndexDomain, IndexDomain], IndexDomain | None],
) -> None:
    """IndexDomain.intersect returns the intersection, or None when disjoint."""
    a, b = case.input
    assert a.intersect(b) == case.expected


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=(IndexDomain.from_shape((10,)), IndexDomain.from_shape((10, 20))),
            msg="different ranks",
            exception_cls=ValueError,
            id="rank-mismatch",
        ),
    ],
    ids=lambda c: c.id,
)
def test_intersect_errors(case: ExpectErr[tuple[IndexDomain, IndexDomain]]) -> None:
    """IndexDomain.intersect raises ValueError on rank mismatch."""
    a, b = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        a.intersect(b)


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(IndexDomain.from_shape((10, 20)), (5, 10)),
            expected=IndexDomain(inclusive_min=(5, 10), exclusive_max=(15, 30)),
            id="positive-offset",
        ),
        Expect(
            input=(IndexDomain(inclusive_min=(10, 20), exclusive_max=(30, 40)), (-10, -20)),
            expected=IndexDomain(inclusive_min=(0, 0), exclusive_max=(20, 20)),
            id="negative-offset",
        ),
        Expect(
            input=(IndexDomain.from_shape((10,)), (0,)),
            expected=IndexDomain.from_shape((10,)),
            id="zero-offset",
        ),
    ],
    ids=lambda c: c.id,
)
def test_translate_success(
    case: Expect[tuple[IndexDomain, tuple[int, ...]], IndexDomain],
) -> None:
    """IndexDomain.translate shifts every coordinate by the offset."""
    domain, offset = case.input
    assert domain.translate(offset) == case.expected


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=(IndexDomain.from_shape((10,)), (1, 2)),
            msg="same length",
            exception_cls=ValueError,
            id="offset-too-long",
        ),
        ExpectErr(
            input=(IndexDomain.from_shape((10, 20)), (1,)),
            msg="same length",
            exception_cls=ValueError,
            id="offset-too-short",
        ),
    ],
    ids=lambda c: c.id,
)
def test_translate_errors(case: ExpectErr[tuple[IndexDomain, tuple[int, ...]]]) -> None:
    """IndexDomain.translate raises when offset length differs from ndim."""
    domain, offset = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        domain.translate(offset)


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=(IndexDomain.from_shape((10, 20)), (slice(2, 8), slice(5, 15))),
            expected=IndexDomain(inclusive_min=(2, 5), exclusive_max=(8, 15)),
            id="2d-slices",
        ),
        Expect(
            input=(IndexDomain.from_shape((10, 20)), (3, slice(None))),
            expected=IndexDomain(inclusive_min=(3, 0), exclusive_max=(4, 20)),
            id="int-and-slice",
        ),
        Expect(
            input=(IndexDomain.from_shape((10, 20, 30)), (slice(1, 5), ...)),
            expected=IndexDomain(inclusive_min=(1, 0, 0), exclusive_max=(5, 20, 30)),
            id="ellipsis-fills-trailing",
        ),
        Expect(
            input=(IndexDomain.from_shape((10,)), (slice(None),)),
            expected=IndexDomain.from_shape((10,)),
            id="slice-none-is-noop",
        ),
        Expect(
            input=(IndexDomain(inclusive_min=(10,), exclusive_max=(20,)), (slice(12, 18),)),
            expected=IndexDomain(inclusive_min=(12,), exclusive_max=(18,)),
            id="non-zero-origin",
        ),
        Expect(
            input=(IndexDomain.from_shape((10,)), (slice(-5, 100),)),
            expected=IndexDomain(inclusive_min=(0,), exclusive_max=(10,)),
            id="clamps-to-domain",
        ),
        Expect(
            input=(IndexDomain.from_shape((10,)), slice(2, 8)),
            expected=IndexDomain(inclusive_min=(2,), exclusive_max=(8,)),
            id="bare-slice-is-wrapped",
        ),
    ],
    ids=lambda c: c.id,
)
def test_narrow_success(case: Expect[tuple[IndexDomain, Any], IndexDomain]) -> None:
    """IndexDomain.narrow applies basic indexing to produce a sub-domain."""
    domain, selection = case.input
    assert domain.narrow(selection) == case.expected


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=(IndexDomain.from_shape((10,)), (10,)),
            msg="out of bounds",
            exception_cls=IndexError,
            id="int-at-upper-bound",
        ),
        ExpectErr(
            input=(IndexDomain(inclusive_min=(5,), exclusive_max=(10,)), (4,)),
            msg="out of bounds",
            exception_cls=IndexError,
            id="int-below-origin",
        ),
        ExpectErr(
            input=(IndexDomain.from_shape((10,)), (1, 2)),
            msg="too many indices",
            exception_cls=IndexError,
            id="too-many-indices",
        ),
        ExpectErr(
            input=(IndexDomain.from_shape((10,)), (slice(0, 10, 2),)),
            msg="step=1",
            exception_cls=IndexError,
            id="non-unit-step",
        ),
    ],
    ids=lambda c: c.id,
)
def test_narrow_errors(case: ExpectErr[tuple[IndexDomain, Any]]) -> None:
    """IndexDomain.narrow raises IndexError on invalid selections."""
    domain, selection = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        domain.narrow(selection)


# ---------------------------------------------------------------------------
# Direct tests for the non-trivial private helper _normalize_selection.
# Public callers (`IndexDomain.narrow` and `selection_to_transform`) exercise
# most branches transitively, but the double-ellipsis guard only triggers on
# inputs no public caller currently constructs. Test it directly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=((slice(2, 8), slice(5, 15)), 2),
            expected=(slice(2, 8), slice(5, 15)),
            id="explicit-slices",
        ),
        Expect(
            input=((3, slice(None)), 2),
            expected=(3, slice(None)),
            id="int-and-slice",
        ),
        Expect(
            input=((..., slice(0, 5)), 3),
            expected=(slice(None), slice(None), slice(0, 5)),
            id="leading-ellipsis-fills",
        ),
        Expect(
            input=((slice(0, 5), ...), 3),
            expected=(slice(0, 5), slice(None), slice(None)),
            id="trailing-ellipsis-fills",
        ),
        Expect(
            input=((slice(2, 8),), 3),
            expected=(slice(2, 8), slice(None), slice(None)),
            id="implicit-trailing-fills",
        ),
        Expect(
            input=(slice(2, 8), 1),
            expected=(slice(2, 8),),
            id="bare-slice-is-wrapped",
        ),
        Expect(
            input=(5, 1),
            expected=(5,),
            id="bare-int-is-wrapped",
        ),
    ],
    ids=lambda c: c.id,
)
def test_normalize_selection_success(
    case: Expect[tuple[Any, int], tuple[int | slice, ...]],
) -> None:
    """_normalize_selection produces a length-ndim tuple of ints/slices."""
    selection, ndim = case.input
    assert _normalize_selection(selection, ndim) == case.expected


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=((..., ..., slice(0, 5)), 3),
            msg="single ellipsis",
            exception_cls=IndexError,
            id="double-ellipsis",
        ),
        ExpectErr(
            input=((1, 2, 3), 2),
            msg="too many indices",
            exception_cls=IndexError,
            id="too-many-indices",
        ),
    ],
    ids=lambda c: c.id,
)
def test_normalize_selection_errors(case: ExpectErr[tuple[Any, int]]) -> None:
    """_normalize_selection rejects double ellipsis and over-long selections."""
    selection, ndim = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        _normalize_selection(selection, ndim)
