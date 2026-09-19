"""Check the extracted contract against the Zarr checkout used for extraction."""

from __future__ import annotations

import importlib
import importlib.util
import inspect

import pytest
from zarr.abc import store as original


def test_storage_package_exists() -> None:
    assert importlib.util.find_spec("zarr_storage") is not None


@pytest.mark.parametrize(
    "name",
    [
        "Store",
        "ByteGetter",
        "ByteSetter",
        "SyncByteGetter",
        "SyncByteSetter",
        "SupportsGetSync",
        "SupportsSetSync",
        "SupportsDeleteSync",
        "SupportsSyncStore",
        "RangeByteRequest",
        "OffsetByteRequest",
        "SuffixByteRequest",
    ],
)
def test_legacy_storage_signatures(name: str) -> None:
    extracted = importlib.import_module("zarr_storage.legacy")
    before = getattr(original, name)
    after = getattr(extracted, name)
    assert before is not after
    assert str(inspect.signature(before)) == str(inspect.signature(after))
    assert getattr(before, "__abstractmethods__", None) == getattr(
        after, "__abstractmethods__", None
    )
    for member_name, member in vars(before).items():
        if isinstance(member, (classmethod, staticmethod)):
            member = member.__func__
            replacement = vars(after)[member_name].__func__
        elif inspect.isfunction(member):
            replacement = vars(after)[member_name]
        else:
            continue
        assert str(inspect.signature(member)) == str(inspect.signature(replacement))
        assert inspect.iscoroutinefunction(member) == inspect.iscoroutinefunction(replacement)
