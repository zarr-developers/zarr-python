"""The absence sentinel for optional metadata-document keys.

The models observe one invariant: `None` in a model always corresponds to a
JSON `null` in the document (a v2 `compressor`/`filters` value, an unnamed
dimension inside `dimension_names`), and `UNSET` always means the document
key is absent. The two are never interchangeable: for keys where the spec
gives `null` no meaning (`dimension_names` itself, `consolidated_metadata`),
a model `None` would invite serializing an invalid `key: null` spelling,
while `UNSET` cannot leak into a document at all.

Check with identity: `if model.dimension_names is UNSET: ...`.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, Literal


class UnsetType(Enum):
    """The type of `UNSET`; use in annotations as `T | UnsetType`."""

    UNSET = "UNSET"

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final[Literal[UnsetType.UNSET]] = UnsetType.UNSET
"""Marks a metadata-document key as absent. Test with `is UNSET`."""
