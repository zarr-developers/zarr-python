"""One document in, one canonical document or one report of why not.

`canonicalize_array_metadata_v3` takes a *syntactically* valid document --
one the model layer has already accepted, so every member is present and
typed as its TypedDict declares -- and answers with
`Canonical[T] | Invalid`: either the same document in canonical form or
every reason it is not semantically valid. Testing the literal `valid`
field narrows to one or the other.

Canonical means the simplest spelling with the same meaning, and each
entity decides that for itself in its own `canonical`: an entity whose
configuration carries nothing collapses to its bare name, `blosc` drops a
`typesize` that `shuffle` renders ignored, a rectilinear dimension's chunk
sizes run-length encode. This module only collects the answers, and the
fields no entity owns -- `dimension_names` of nothing but nulls says what
omitting the field says.

Two properties are worth holding on to, and
`tests/rules/test_canonical.py` asserts both: canonicalizing twice
changes nothing further, and canonicalizing never changes a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, cast

from zarr_metadata.model._array import ZarrV3ArrayMetadata
from zarr_metadata.model._validation import arrays_to_tuples
from zarr_metadata.rules._documents import validate_array_metadata_v3
from zarr_metadata.v3._document import read_array_v3
from zarr_metadata.v3._entity import MetadataEntity
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.model._validation import ValidationProblem
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON

DocumentT = TypeVar("DocumentT")


@dataclass(frozen=True, slots=True)
class Canonical(Generic[DocumentT]):
    """A semantically valid document, in its simplest equivalent spelling."""

    document: DocumentT
    valid: Literal[True] = True


@dataclass(frozen=True, slots=True)
class Invalid:
    """Every reason a document is not semantically valid; never empty."""

    problems: tuple[ValidationProblem, ...]
    valid: Literal[False] = False

    def __post_init__(self) -> None:
        if len(self.problems) == 0:
            msg = "Invalid requires at least one validation problem"
            raise ValueError(msg)


def _canonical_document(document: Mapping[str, object], context: Context) -> dict[str, object]:
    """Each entity in its own canonical spelling, and the rest as written."""
    array, _ = read_array_v3(document, context)
    out = dict(document)
    for key in ("data_type", "chunk_grid", "chunk_key_encoding"):
        entity = getattr(array, key)
        if isinstance(entity, MetadataEntity):
            out[key] = entity.canonical().to_json()
    if "codecs" in out:
        out["codecs"] = tuple(
            codec.canonical().to_json() if isinstance(codec, MetadataEntity) else codec.json
            for codec in array.codecs
        )
    names = out.get("dimension_names")
    if isinstance(names, tuple) and all(
        entry is None for entry in cast("tuple[object, ...]", names)
    ):
        # Every dimension unnamed says what saying nothing says.
        del out["dimension_names"]
    return out


def canonicalize_array_metadata_v3(
    document: ZarrV3ArrayMetadataJSON, *, context: Context = CORE_AND_EXTENSIONS
) -> Canonical[ZarrV3ArrayMetadataJSON] | Invalid:
    """`document` in canonical form, or every reason it is not valid.

    Expects a document the model layer has already accepted. Passing one
    it has not is not an error -- the semantic problems are reported the
    same way -- but the structural problems come back too, and the result
    is `Invalid` rather than a canonical document.
    """
    normalized = cast("ZarrV3ArrayMetadataJSON", arrays_to_tuples(document))
    problems = validate_array_metadata_v3(normalized, context=context)
    if len(problems) != 0:
        return Invalid(problems)
    # Normalized first, so a document spelled with JSON arrays reaches the
    # same fixpoint as the tuple spelling. It did not: the per-field
    # simplifications test for `tuple`, and the validator was normalizing
    # on a copy the canonicalizer never saw.
    canonical = _canonical_document(normalized, context)
    # The model layer's round trip normalizes the fields no entity owns.
    return Canonical(ZarrV3ArrayMetadata.from_json(canonical).to_json())


__all__ = [
    "Canonical",
    "Invalid",
    "canonicalize_array_metadata_v3",
]
