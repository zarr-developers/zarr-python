"""Drift-prevention tests for Partial* TypedDict variants.

Each *Partial TypedDict in the package must declare the same fields
(with the same annotations) and the same extra_items setting as its
full counterpart. The only intentional difference is total=False
(i.e. every field becomes NotRequired). This test enforces that
invariant so adding a field to the full type without mirroring it
on the partial fails CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from zarr_metadata.v2.array import ArrayMetadataV2, ArrayMetadataV2Partial
from zarr_metadata.v2.group import GroupMetadataV2, GroupMetadataV2Partial
from zarr_metadata.v3.array import ArrayMetadataV3, ArrayMetadataV3Partial
from zarr_metadata.v3.group import GroupMetadataV3, GroupMetadataV3Partial

# (full, partial) pairs to check. Add new pairs here as more are introduced.
PAIRS: list[tuple[type, type]] = [
    (ArrayMetadataV3, ArrayMetadataV3Partial),
    (GroupMetadataV3, GroupMetadataV3Partial),
    (ArrayMetadataV2, ArrayMetadataV2Partial),
    (GroupMetadataV2, GroupMetadataV2Partial),
]


@pytest.mark.parametrize(("full", "partial"), PAIRS, ids=lambda p: p.__name__)
def test_partial_matches_full(full: Any, partial: Any) -> None:
    """Partial TypedDict has identical fields and extra_items, only total differs."""
    assert full.__annotations__ == partial.__annotations__, (
        f"{partial.__name__} fields drifted from {full.__name__}: "
        f"full={set(full.__annotations__)}, partial={set(partial.__annotations__)}"
    )
    assert getattr(full, "__extra_items__", None) == getattr(partial, "__extra_items__", None), (
        f"{partial.__name__} extra_items differs from {full.__name__}"
    )
    assert full.__total__ is True, f"{full.__name__} must be declared with total=True (default)"
    assert partial.__total__ is False, f"{partial.__name__} must be declared with total=False"
