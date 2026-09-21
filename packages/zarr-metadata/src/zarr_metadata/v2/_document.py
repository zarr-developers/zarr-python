"""What a v2 array document can get wrong beyond its shape.

Deliberately small: the one cross-field constraint the package interprets
is that `chunks` and `shape` agree on dimensionality. v2 has no extension
mechanism, so there are no entities to ask -- this is the whole of it.
Fill-value/dtype consistency for v2 (NumPy dtype strings, base64 fills
for bytes dtypes) is a known follow-up, tracked in the package docs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem

if TYPE_CHECKING:
    from collections.abc import Mapping


def _as_sequence(value: object) -> tuple[object, ...] | None:
    """`value` as a tuple if it is a JSON array, else None."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        return None
    return tuple(cast("Sequence[object]", value))


def array_problems_v2(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """Every semantic problem in a v2 array document."""
    shape = _as_sequence(document.get("shape"))
    chunks = _as_sequence(document.get("chunks"))
    if shape is None or chunks is None or len(shape) == len(chunks):
        return ()
    return (
        ValidationProblem(
            ("chunks",),
            "expected the same number of dimensions as shape",
            "invalid_value",
        ),
    )


__all__ = [
    "array_problems_v2",
]
