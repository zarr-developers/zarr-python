"""`CORE` and `CORE_AND_EXTENSIONS`, the v3 scopes' names before they ended in `_CONTEXT`, which still work with a warning."""

from __future__ import annotations

import importlib

import pytest

from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS_CONTEXT, CORE_CONTEXT, ZarrV3Context


@pytest.mark.parametrize("module", ["zarr_metadata.v3", "zarr_metadata.v3.definition"])
@pytest.mark.parametrize(
    ("old", "new", "scope"),
    [
        ("CORE", "CORE_CONTEXT", CORE_CONTEXT),
        ("CORE_AND_EXTENSIONS", "CORE_AND_EXTENSIONS_CONTEXT", CORE_AND_EXTENSIONS_CONTEXT),
    ],
)
def test_an_old_name_is_the_same_scope_with_a_deprecation_warning(
    module: str, old: str, new: str, scope: ZarrV3Context
) -> None:
    with pytest.warns(DeprecationWarning, match=f"`{old}` is deprecated.*use `{new}`") as caught:
        found = getattr(importlib.import_module(module), old)
    assert found is scope
    # The warning points at the code that used the old name, not at this package.
    assert caught[0].filename == __file__


@pytest.mark.parametrize("module", ["zarr_metadata.v3", "zarr_metadata.v3.definition"])
def test_error_a_name_that_is_neither_old_nor_new(module: str) -> None:
    with pytest.raises(AttributeError, match="has no attribute 'CORE_V3'"):
        importlib.import_module(module).CORE_V3  # noqa: B018
