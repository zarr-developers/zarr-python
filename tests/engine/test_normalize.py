from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

from zarr.core.engine import (
    apply_post_index,
    normalize_basic,
    normalize_coordinate,
    normalize_orthogonal,
    strip_squeeze,
)
from zarr.core.engine._normalize import _Squeeze

if TYPE_CHECKING:
    from zarr.abc.engine import Region

SHAPE = (10, 9)
ARR = np.arange(90).reshape(SHAPE)


def _read_box(region: Region) -> npt.NDArray[np.int_]:
    """Simulate an engine read: the ndim-preserving box."""
    return ARR[tuple(slice(s, e) for s, e in zip(region.start, region.end_exclusive, strict=True))]


@pytest.mark.parametrize(
    "sel",
    [
        (slice(2, 7), slice(0, 9)),
        (slice(None), slice(3, 4)),
        (slice(8, 2, -2), slice(None)),
        (slice(1, 8, 3), slice(2, 9, 2)),
        (3, slice(None)),
        (-1, -2),
        (Ellipsis, 4),
        slice(5),
        Ellipsis,
        (slice(4, 4), slice(None)),  # empty
    ],
)
def test_normalize_basic_matches_numpy(sel: Any) -> None:
    region, post = normalize_basic(sel, SHAPE)
    np.testing.assert_array_equal(apply_post_index(_read_box(region), post), ARR[sel])


@pytest.mark.parametrize(
    "sel",
    [
        (np.array([1, 4, 7]), np.array([0, 8])),
        (np.array([7, 1, 4]), slice(2, 6)),  # unordered axis indices
        (np.array([3, 3]), np.array([5, 5])),  # repeats
        (slice(1, 9, 2), np.array([2])),
        (np.array([0]), 4),  # int axis
    ],
)
def test_normalize_orthogonal_matches_numpy_oindex(sel: Any) -> None:
    from zarr.core.indexing import oindex as _reference_oindex

    region, post = normalize_orthogonal(sel, SHAPE)
    # `zarr.core.indexing.oindex` is the existing, well-tested reference
    # implementation of orthogonal indexing (np.ix_ outer indexing with
    # integer axes dropped afterwards) -- use it as the ground truth rather
    # than reimplementing it by hand.
    expected = _reference_oindex(ARR, sel if isinstance(sel, tuple) else (sel,))
    np.testing.assert_array_equal(apply_post_index(_read_box(region), post), expected)


def test_normalize_coordinate_matches_numpy_vindex() -> None:
    coords = (np.array([9, 0, 3, 3]), np.array([8, 0, 2, 2]))
    region, post = normalize_coordinate(coords, SHAPE)
    np.testing.assert_array_equal(apply_post_index(_read_box(region), post), ARR[coords])


def test_normalize_basic_rejects_fancy() -> None:
    with pytest.raises(TypeError):
        normalize_basic((np.array([1, 2]), slice(None)), SHAPE)  # type: ignore[arg-type]


def test_normalize_basic_rejects_out_of_bounds_int() -> None:
    with pytest.raises(IndexError):
        normalize_basic((10, slice(None)), SHAPE)


def test_normalize_coordinate_rejects_out_of_bounds() -> None:
    with pytest.raises(IndexError):
        normalize_coordinate((np.array([10]), np.array([0])), SHAPE)


def test_normalize_orthogonal_rejects_boolean_ndim_mismatch() -> None:
    with pytest.raises(IndexError):
        normalize_orthogonal((np.zeros((2, 2), dtype=bool), slice(None)), SHAPE)


def test_strip_squeeze_removes_trailing_marker() -> None:
    post_with_marker = (slice(None), np.array([0]), _Squeeze((1,)))
    post_without = (slice(None), np.array([0]))
    stripped = strip_squeeze(post_with_marker)
    assert len(stripped) == len(post_without)
    assert stripped[0] == post_without[0]
    np.testing.assert_array_equal(stripped[1], post_without[1])


def test_strip_squeeze_identity_when_no_marker() -> None:
    post = (slice(None), slice(2, 4))
    assert strip_squeeze(post) == post


def test_normalize_orthogonal_list_of_bools_matches_array_bool_mask() -> None:
    """List of booleans should produce the same result as np.ndarray bool mask."""
    from zarr.core.indexing import oindex as _reference_oindex

    # Test case from the defect report: list of 10 booleans for axis 0
    bool_list = [True, False, True] + [False] * 7
    bool_array = np.array(bool_list, dtype=bool)

    # Both should produce the same result when used as selectors
    region_list, post_list = normalize_orthogonal((bool_list, slice(None)), SHAPE)
    region_array, post_array = normalize_orthogonal((bool_array, slice(None)), SHAPE)

    # Reference implementation using zarr.core.indexing.oindex
    expected = _reference_oindex(ARR, (bool_array, slice(None)))

    # List of bools should match the array bool mask result
    result_list = apply_post_index(_read_box(region_list), post_list)
    result_array = apply_post_index(_read_box(region_array), post_array)

    np.testing.assert_array_equal(result_list, expected)
    np.testing.assert_array_equal(result_array, expected)


def test_normalize_orthogonal_wrong_length_list_of_bools_raises_error() -> None:
    """Wrong-length list of booleans should raise IndexError like ndarray."""
    wrong_length_bool_list = [True, False, True]  # Length 3, but axis 0 has size 10

    with pytest.raises(IndexError, match="boolean index for axis 0 must be 1-d with length 10"):
        normalize_orthogonal((wrong_length_bool_list, slice(None)), SHAPE)
