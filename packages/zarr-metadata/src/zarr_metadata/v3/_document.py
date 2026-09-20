"""A whole v3 array document, read as entities and judged as a whole.

The document-level check is a composition of the entities' own checks,
not a second implementation of them. It does three things in order:

1. read every extension point in a scope, which is type-space;
2. ask each entity what is wrong with its own values;
3. ask the questions that span fields -- the fill value against the data
   type, the grid against the shape, the pipeline against the array --
   each by handing an entity the part of the document it needs.

Nothing here knows what `blosc` or `int32` or `rectilinear` is. A new
extension is a class and a registry entry, and this module does not
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._chain import chain_problems
from zarr_metadata.v3._entity import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
    STORAGE_TRANSFORMERS,
    ChunkGridEntity,
    DataTypeEntity,
    ExtensionPointField,
    MetadataEntity,
    within,
)
from zarr_metadata.v3._parts import ArrayParts, ChunkGrid

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr_metadata.v3._registry import Context


@dataclass(frozen=True, slots=True)
class ArrayDocumentV3:
    """A v3 array document with its extension points read as entities.

    A field holds the value untouched where its name was out of scope, so
    an unmodelled extension survives the reading and is simply not judged.
    """

    document: Mapping[str, object]
    data_type: MetadataEntity | object
    chunk_grid: MetadataEntity | object
    chunk_key_encoding: MetadataEntity | object
    codecs: tuple[MetadataEntity | object, ...]
    storage_transformers: tuple[MetadataEntity | object, ...]

    @property
    def parts(self) -> ArrayParts:
        """The array the codec pipeline is handed."""
        shape = self.document.get("shape")
        # A grid out of scope still divides an array of some rank, and the
        # shape is what pins it -- which is enough to catch a shard whose
        # inner chunk has the wrong number of dimensions.
        grid = (
            self.chunk_grid.grid(shape)
            if isinstance(self.chunk_grid, ChunkGridEntity)
            else ChunkGrid.unreadable(shape)
        )
        return ArrayParts(
            grid, self.data_type if isinstance(self.data_type, DataTypeEntity) else None
        )


# The three extension points a document names once, and the field each is
# named in. `codecs` is the fourth and holds a list, so it is separate.
_SINGLE_FIELDS: Final[tuple[tuple[ExtensionPointField, str], ...]] = (
    (DATA_TYPE, "data_type"),
    (CHUNK_GRID, "chunk_grid"),
    (CHUNK_KEY_ENCODING, "chunk_key_encoding"),
)

# The two the document names as a list. Nothing models a storage
# transformer yet, so nothing is judged there today -- but the extension
# point is registerable, and a registered one has to be reached.
_SEQUENCE_FIELDS: Final[tuple[tuple[ExtensionPointField, str], ...]] = (
    (CODECS, "codecs"),
    (STORAGE_TRANSFORMERS, "storage_transformers"),
)


def read_array_v3(
    document: Mapping[str, object], context: Context
) -> tuple[ArrayDocumentV3, tuple[ValidationProblem, ...]]:
    """`document`'s extension points, read in `context`.

    Type-space only: what comes back is well-typed by construction, and
    the problems are the reasons some of it is not an entity.
    """
    read: dict[str, MetadataEntity | object] = {}
    problems: list[ValidationProblem] = []
    for field, key in _SINGLE_FIELDS:
        value = document.get(key)
        if value is None:
            read[key] = None
            continue
        entity, found = context.coerce(field, value, (key,), envelope_judged=True)
        read[key] = entity
        problems.extend(found)
    sequences: dict[str, tuple[MetadataEntity | object, ...]] = {}
    for field, key in _SEQUENCE_FIELDS:
        read_entries: list[MetadataEntity | object] = []
        entries = document.get(key)
        if isinstance(entries, (list, tuple)):
            for index, entry in enumerate(cast("Sequence[object]", entries)):
                entity, found = context.coerce(field, entry, (key, index), envelope_judged=True)
                read_entries.append(entity)
                problems.extend(found)
        sequences[key] = tuple(read_entries)
    return (
        ArrayDocumentV3(
            document=document,
            data_type=read["data_type"],
            chunk_grid=read["chunk_grid"],
            chunk_key_encoding=read["chunk_key_encoding"],
            codecs=sequences["codecs"],
            storage_transformers=sequences["storage_transformers"],
        ),
        tuple(problems),
    )


def _entity_problems(array: ArrayDocumentV3) -> tuple[ValidationProblem, ...]:
    """What each entity says is wrong with its own values."""
    found: list[ValidationProblem] = []
    for _, key in _SINGLE_FIELDS:
        entity = getattr(array, key)
        if isinstance(entity, MetadataEntity):
            found.extend(within((key,), entity.problems()))
    for _, key in _SEQUENCE_FIELDS:
        for index, entity in enumerate(cast("tuple[object, ...]", getattr(array, key))):
            if isinstance(entity, MetadataEntity):
                found.extend(within((key, index), entity.problems()))
    return tuple(found)


def _fill_value_problems(array: ArrayDocumentV3) -> tuple[ValidationProblem, ...]:
    """The fill value, judged by the data type it fills."""
    if not isinstance(array.data_type, DataTypeEntity) or "fill_value" not in array.document:
        return ()
    return array.data_type.fill_value_problems(array.document["fill_value"], ("fill_value",))


def _dimension_names_problems(array: ArrayDocumentV3) -> tuple[ValidationProblem, ...]:
    """One name per dimension, if names are given at all."""
    names = array.document.get("dimension_names")
    shape = array.document.get("shape")
    if not isinstance(names, (list, tuple)) or not isinstance(shape, (list, tuple)):
        return ()
    given = len(cast("Sequence[object]", names))
    rank = len(cast("Sequence[object]", shape))
    if given == rank:
        return ()
    return (
        ValidationProblem(
            ("dimension_names",),
            f"dimension_names has {given} entries but shape has {rank} dimensions",
            "invalid_value",
        ),
    )


def _grid_problems(array: ArrayDocumentV3) -> tuple[ValidationProblem, ...]:
    """The chunk grid, judged against the array it divides."""
    if not isinstance(array.chunk_grid, ChunkGridEntity):
        return ()
    return within(("chunk_grid",), array.chunk_grid.shape_problems(array.document.get("shape")))


def array_problems_v3(
    document: Mapping[str, object], context: Context
) -> tuple[ValidationProblem, ...]:
    """Every semantic problem in `document`, read in `context`.

    Expects a document the model layer has already accepted, so every
    member is present and typed as its TypedDict declares.
    """
    array, problems = read_array_v3(document, context)
    return (
        *problems,
        *_entity_problems(array),
        *_fill_value_problems(array),
        *_grid_problems(array),
        *_dimension_names_problems(array),
        *chain_problems(array.codecs, array.parts, ("codecs",)),
    )


__all__ = [
    "ArrayDocumentV3",
    "array_problems_v3",
    "read_array_v3",
]
