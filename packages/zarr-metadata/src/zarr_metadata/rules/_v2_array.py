"""Composition rules for v2 array metadata documents.

The v2 rule set is deliberately small today: the one cross-field
constraint the package interprets is that `chunks` and `shape` agree on
dimensionality. Fill-value/dtype consistency for v2 (NumPy dtype strings,
base64 fills for bytes dtypes) is a known follow-up, tracked in the
package docs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from zarr_metadata.model._validation import (
    ARRAY_METADATA_STANDARD_KEYS_V2,
    ValidationProblem,
)
from zarr_metadata.rules._engine import Rule, as_sequence
from zarr_metadata.rules._registry import document_rule, document_rules, register_document_type

if TYPE_CHECKING:
    from collections.abc import Mapping


ZARR_V2_ARRAY = "zarr_v2_array"
"""Document-type key under which this module's rules are registered."""

register_document_type(ZARR_V2_ARRAY, ARRAY_METADATA_STANDARD_KEYS_V2)


@document_rule(ZARR_V2_ARRAY, frozenset({"shape", "chunks"}))
def check_chunks_match_shape(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """`chunks` must have one entry per dimension of `shape`."""
    shape = as_sequence(document["shape"])
    chunks = as_sequence(document["chunks"])
    if shape is None or chunks is None or len(shape) == len(chunks):
        return ()
    return (
        ValidationProblem(
            ("chunks",),
            "expected the same number of dimensions as shape",
            "invalid_value",
        ),
    )


ZARR_V2_ARRAY_RULES: Final[tuple[Rule, ...]] = document_rules(ZARR_V2_ARRAY)
"""The composition rule set for v2 array metadata documents."""


__all__ = [
    "ZARR_V2_ARRAY",
    "ZARR_V2_ARRAY_RULES",
]
