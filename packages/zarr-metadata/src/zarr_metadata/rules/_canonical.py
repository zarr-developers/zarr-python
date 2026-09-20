"""One document in, one canonical document or one report of why not.

`canonicalize_array_metadata_v3` takes a *syntactically* valid document —
one the model layer has already accepted, so every member is present and
typed as its TypedDict declares — and answers with `Canonical[T] | Invalid`: either the same document in
canonical form or every reason it is not semantically valid. Testing the
literal `valid` field narrows to one or the other.

Canonical means the simplest spelling with the same meaning, decided per
metadata variety:

- an entity whose configuration carries nothing collapses to its bare
  name, and a `must_understand` of `true` (the default) is dropped while
  an explicit `false` is kept, because that one says something. This part
  the model layer already performs, so it is delegated rather than
  reimplemented.
- `blosc` drops a `typesize` that `shuffle: "noshuffle"` renders ignored.
- a rectilinear dimension's chunk sizes run-length encode, because that
  is the spelling that does not grow with the number of chunks.
- `dimension_names` of nothing but nulls says what omitting the field
  says.

Two properties are worth holding on to, and
`tests/rules/test_canonical.py` asserts both: canonicalizing twice
changes nothing further, and canonicalizing never changes a verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, cast

from zarr_metadata.model._array import ZarrV3ArrayMetadata
from zarr_metadata.rules._documents import validate_array_metadata_v3
from zarr_metadata.v3._shape import entity_name
from zarr_metadata.v3.chunk_grid.rectilinear import (
    RECTILINEAR_CHUNK_GRID_NAME,
    canonical_chunk_shapes,
)
from zarr_metadata.v3.codec.blosc import BLOSC_CODEC_NAME
from zarr_metadata.v3.codec.blosc import canonical_configuration as canonical_blosc

if TYPE_CHECKING:
    from zarr_metadata.model._validation import ValidationProblem
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
    from zarr_metadata.v3.chunk_grid.rectilinear import RectilinearDimSpec

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


def _canonical_entity(value: object) -> object:
    """One entity's configuration in its simplest equivalent form.

    Only varieties with something to say appear here; everything else is
    handed to the generic collapse unchanged.
    """
    original: object = value
    name = entity_name(value)
    if name is None or not isinstance(value, Mapping):
        return original
    entry: Mapping[str, object] = cast("Mapping[str, object]", value)
    configuration = entry.get("configuration")
    if not isinstance(configuration, Mapping):
        return original
    members: Mapping[str, object] = cast("Mapping[str, object]", configuration)
    if name == BLOSC_CODEC_NAME:
        members = canonical_blosc(members)
    elif name == RECTILINEAR_CHUNK_GRID_NAME:
        shapes = members.get("chunk_shapes")
        if isinstance(shapes, tuple):
            specs = cast("tuple[RectilinearDimSpec, ...]", shapes)
            members = {**members, "chunk_shapes": canonical_chunk_shapes(specs)}
    if members is configuration:
        return original
    return {**entry, "configuration": members}


def _canonical_document(document: Mapping[str, object]) -> dict[str, object]:
    """Per-variety canonicalization, before the generic collapse."""
    out = dict(document)
    for field in ("chunk_grid", "chunk_key_encoding", "data_type"):
        if field in out:
            out[field] = _canonical_entity(out[field])
    codecs = out.get("codecs")
    if isinstance(codecs, tuple):
        entries = cast("tuple[object, ...]", codecs)
        out["codecs"] = tuple(_canonical_entity(codec) for codec in entries)
    names = out.get("dimension_names")
    if isinstance(names, tuple) and all(
        entry is None for entry in cast("tuple[object, ...]", names)
    ):
        # Every dimension unnamed says what saying nothing says.
        del out["dimension_names"]
    return out


def canonicalize_array_metadata_v3(
    document: ZarrV3ArrayMetadataJSON,
) -> Canonical[ZarrV3ArrayMetadataJSON] | Invalid:
    """`document` in canonical form, or every reason it is not valid.

    Expects a document the model layer has already accepted. Passing one
    it has not is not an error — the composition problems are reported the
    same way — but the structural problems come back too, and the result
    is `Invalid` rather than a canonical document.
    """
    problems = validate_array_metadata_v3(document)
    if len(problems) != 0:
        return Invalid(problems)
    canonical = _canonical_document(document)
    # The generic collapse — shorthand names, defaulted `must_understand` —
    # is what the model layer's round trip already performs.
    return Canonical(ZarrV3ArrayMetadata.from_json(canonical).to_json())


__all__ = [
    "Canonical",
    "Invalid",
    "canonicalize_array_metadata_v3",
]
