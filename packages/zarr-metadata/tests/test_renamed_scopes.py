"""`CORE`, `CORE_AND_EXTENSIONS` and `CORE_V2`, the scopes' names before they ended in `_CONTEXT`, which still work with a warning."""

from __future__ import annotations

import importlib

import pytest

from zarr_metadata.v2.definition import CORE_V2_CONTEXT
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS_CONTEXT, CORE_CONTEXT

_V3 = ("zarr_metadata.v3", "zarr_metadata.v3.definition")
_V2 = ("zarr_metadata.v2", "zarr_metadata.v2.definition")


@pytest.mark.parametrize(
    ("module", "old", "new", "scope"),
    [
        *((module, "CORE", "CORE_CONTEXT", CORE_CONTEXT) for module in _V3),
        *(
            (
                module,
                "CORE_AND_EXTENSIONS",
                "CORE_AND_EXTENSIONS_CONTEXT",
                CORE_AND_EXTENSIONS_CONTEXT,
            )
            for module in _V3
        ),
        *((module, "CORE_V2", "CORE_V2_CONTEXT", CORE_V2_CONTEXT) for module in _V2),
    ],
)
def test_an_old_name_is_the_same_scope_with_a_deprecation_warning(
    module: str, old: str, new: str, scope: object
) -> None:
    with pytest.warns(DeprecationWarning, match=f"`{old}` is deprecated.*use `{new}`") as caught:
        found = getattr(importlib.import_module(module), old)
    assert found is scope
    # The warning points at the code that used the old name, not at this package.
    assert caught[0].filename == __file__


@pytest.mark.parametrize("module", [*_V3, *_V2])
def test_error_a_name_that_is_neither_old_nor_new(module: str) -> None:
    with pytest.raises(AttributeError, match="has no attribute 'CORE_V3'"):
        importlib.import_module(module).CORE_V3  # noqa: B018
