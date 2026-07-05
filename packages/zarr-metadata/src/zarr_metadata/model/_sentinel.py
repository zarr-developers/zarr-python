"""The absence sentinel for optional metadata-document keys.

The models observe one invariant: `None` in a model always corresponds to a
JSON `null` in the document (a v2 `compressor`/`filters` value, an unnamed
dimension inside `dimension_names`), and `UNSET` always means the document
key is absent. The two are never interchangeable, so a model value can never leak
into a document as a spelling the writer did not intend.

Check with identity: `if model.dimension_names is UNSET: ...`.

Implementation note: `typing_extensions.Sentinel` (PEP 661) is the intended
spelling, but pyright (1.1.411) degrades a Sentinel to `Unknown` in dataclass
FIELD annotations (function signatures work), which would force suppressions
under this package's strict gate at every use site. The single-member enum
gives the same identity semantics with exact `Literal` narrowing; switch to
`Sentinel` once pyright supports it in dataclass fields.
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
