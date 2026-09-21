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

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeVar, cast

from zarr_metadata.model._validation import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
)
from zarr_metadata.model._validation import (
    validate_array_metadata_v3 as validate_array_metadata_v3_structure,
)
from zarr_metadata.v3._chain import chain_problems
from zarr_metadata.v3._entity import (
    ChunkGridEntity,
    ChunkKeyEncodingEntity,
    CodecEntity,
    DataTypeEntity,
    MetadataEntity,
    Opaque,
    StorageTransformerEntity,
    canonicalized,
    within,
    written,
)
from zarr_metadata.v3._parts import ArrayParts, ChunkGrid
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata.v3._entity import Loc


_EntityT = TypeVar("_EntityT", bound=MetadataEntity)


@dataclass(frozen=True, slots=True)
class ArrayDocumentV3:
    """A v3 array document with its extension points read as entities.

    A field that could not be read holds an `Opaque`, which carries the
    JSON the document wrote and says whether the name was out of scope --
    an extension this reader does not model, which is not an error -- or
    claimed and refused. Both are narrowable: every field is an exhaustive
    two-case union.
    """

    document: Mapping[str, object]
    data_type: DataTypeEntity | Opaque
    chunk_grid: ChunkGridEntity | Opaque
    chunk_key_encoding: ChunkKeyEncodingEntity | Opaque
    codecs: tuple[CodecEntity | Opaque, ...]
    storage_transformers: tuple[StorageTransformerEntity | Opaque, ...]

    def problems(self) -> tuple[ValidationProblem, ...]:
        """Every semantic problem this document has, once it has been read.

        The type-space problems are `read_array_v3`'s, because they are
        the reasons some of this is `Opaque` rather than an entity.
        """
        # No per-entity value problems: an entity exists only if its own
        # values are allowed, so `read_array_v3` has already reported any.
        return (
            *_fill_value_problems(self),
            *_grid_problems(self),
            *_dimension_names_problems(self),
            *chain_problems(self.codecs, self.parts, ("codecs",)),
        )

    def canonical(self) -> ArrayDocumentV3:
        """This document in the simplest form that means the same thing.

        Each entity in its own canonical form, and the one rule that is
        the document's own: `dimension_names` of nothing but nulls says
        what omitting the field says. A *transformation*, asked for by
        `canonicalize_array_metadata_v3`; `to_json` does not apply it.
        """
        document = dict(self.document)
        names = document.get("dimension_names")
        if isinstance(names, tuple) and all(
            entry is None for entry in cast("tuple[object, ...]", names)
        ):
            del document["dimension_names"]
        return replace(
            self,
            document=document,
            data_type=canonicalized(self.data_type),
            chunk_grid=canonicalized(self.chunk_grid),
            chunk_key_encoding=canonicalized(self.chunk_key_encoding),
            codecs=tuple(canonicalized(codec) for codec in self.codecs),
            storage_transformers=tuple(canonicalized(entry) for entry in self.storage_transformers),
        )

    def to_json(self) -> dict[str, object]:
        """The document as it would be written: every entity in its JSON form.

        Faithful to what was read, member for member; the fields that
        are not extension points come back exactly as the document had
        them, and a field the document did not have is not invented.
        Ask `canonical` first for the simplest equivalent spelling.
        """
        rendered: dict[str, object] = {
            "data_type": written(self.data_type),
            "chunk_grid": written(self.chunk_grid),
            "chunk_key_encoding": written(self.chunk_key_encoding),
            "codecs": tuple(written(codec) for codec in self.codecs),
            "storage_transformers": tuple(written(entry) for entry in self.storage_transformers),
        }
        return {
            **self.document,
            **{key: value for key, value in rendered.items() if key in self.document},
        }

    @classmethod
    def from_json(cls, value: object, *, context: Context = CORE_AND_EXTENSIONS) -> ArrayDocumentV3:
        """A v3 array document read into entities, or raise.

        The reader's front door, and the one entry point that fails fast:
        one call, and either every extension point is read or a single
        `MetadataValidationError` carries every reason it is not --
        structural and semantic together. Use `validate_array_metadata_v3`
        instead when you want the problems as data.

        A name this `context` does not model is *not* a failure. It comes
        back as an `Opaque` marked `out_of_scope`, because a document may
        legitimately use an extension this reader does not know, and
        refusing it would make openness unimplementable. What fails is
        metadata that is wrong, not metadata that is unfamiliar.
        """
        normalized = arrays_to_tuples(value)
        problems = validate_array_metadata_v3_structure(normalized)
        if isinstance(normalized, Mapping) and len(problems) == 0:
            document = cast("Mapping[str, object]", normalized)
            array, found = read_array_v3(document, context)
            problems = (*found, *array.problems())
            if len(problems) == 0:
                return array
        if len(problems) == 0:  # pragma: no cover - a non-mapping always has problems
            problems = (ValidationProblem((), "expected a v3 array document", "invalid_type"),)
        raise MetadataValidationError(problems)

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


def _read_one(
    context: Context, kind: type[_EntityT], document: Mapping[str, object], key: str
) -> tuple[_EntityT | Opaque, tuple[ValidationProblem, ...]]:
    """The entity of `kind` the document names at `key`; an `Opaque` if it names none."""
    value = document.get(key)
    if value is None:
        return Opaque(None, "invalid"), ()
    return context.coerce(kind, value, (key,), envelope_judged=True)


def _read_each(
    context: Context, kind: type[_EntityT], document: Mapping[str, object], key: str
) -> tuple[tuple[_EntityT | Opaque, ...], tuple[ValidationProblem, ...]]:
    """The entities of `kind` the document lists at `key`, in order."""
    entries = document.get(key)
    if not isinstance(entries, (list, tuple)):
        return (), ()
    read: list[_EntityT | Opaque] = []
    problems: list[ValidationProblem] = []
    for index, entry in enumerate(cast("Sequence[object]", entries)):
        entity, found = context.coerce(kind, entry, (key, index), envelope_judged=True)
        read.append(entity)
        problems.extend(found)
    return tuple(read), tuple(problems)


def read_array_v3(
    document: Mapping[str, object], context: Context
) -> tuple[ArrayDocumentV3, tuple[ValidationProblem, ...]]:
    """`document`'s extension points, read in `context`.

    The one place that knows which of a document's fields holds which
    kind of entity. Type-space only: what comes back is well-typed by
    construction, and the problems are the reasons some of it is not an
    entity.
    """
    data_type, found_1 = _read_one(context, DataTypeEntity, document, "data_type")
    chunk_grid, found_2 = _read_one(context, ChunkGridEntity, document, "chunk_grid")
    encoding, found_3 = _read_one(context, ChunkKeyEncodingEntity, document, "chunk_key_encoding")
    codecs, found_4 = _read_each(context, CodecEntity, document, "codecs")
    transformers, found_5 = _read_each(
        context, StorageTransformerEntity, document, "storage_transformers"
    )
    return (
        ArrayDocumentV3(
            document=document,
            data_type=data_type,
            chunk_grid=chunk_grid,
            chunk_key_encoding=encoding,
            codecs=codecs,
            storage_transformers=transformers,
        ),
        (*found_1, *found_2, *found_3, *found_4, *found_5),
    )


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
    return (*problems, *array.problems())


def _prefixed(loc: Loc, problems: Sequence[ValidationProblem]) -> tuple[ValidationProblem, ...]:
    """Re-base every problem's `loc` under `loc`, for a nested document."""
    return tuple(
        ValidationProblem((*loc, *found.loc), found.message, found.kind) for found in problems
    )


def _as_string_mapping(value: object) -> Mapping[str, object] | None:
    """`value` as a string-keyed mapping, or None if it is not one."""
    if not isinstance(value, Mapping):
        return None
    mapping = cast("Mapping[object, object]", value)
    if any(not isinstance(key, str) for key in mapping):
        return None
    return cast("Mapping[str, object]", mapping)


def group_problems_v3(
    document: Mapping[str, object], context: Context = CORE_AND_EXTENSIONS
) -> tuple[ValidationProblem, ...]:
    """Every semantic problem in a v3 group document.

    A group says almost nothing that can be wrong on its own. The one
    thing it can carry is consolidated metadata -- the child documents of
    a whole subtree, inline -- and each of those is judged exactly as it
    would be standing alone, at its own path. `consolidated_metadata` is
    not a declared member of the group TypedDict: the spec grandfathers
    it as a convention that "lacks the name member required of extension
    objects".
    """
    if "consolidated_metadata" not in document:
        return ()
    return consolidated_entries_problems(
        document["consolidated_metadata"], ("consolidated_metadata",), context
    )


def consolidated_entries_problems(
    value: object, loc: Loc = (), context: Context = CORE_AND_EXTENSIONS
) -> tuple[ValidationProblem, ...]:
    """Semantic problems in an inline consolidated envelope's children.

    Structural validity of the envelope and its entries is the model
    layer's job; an entry that is not interpretable as a node document
    declines in its favour.
    """
    consolidated = _as_string_mapping(value)
    if consolidated is None:
        return ()
    metadata = _as_string_mapping(consolidated.get("metadata"))
    if metadata is None:
        return ()
    problems: list[ValidationProblem] = []
    for path, entry in metadata.items():
        node = _as_string_mapping(entry)
        if node is None:
            continue
        entry_loc = (*loc, "metadata", path)
        node_type = node.get("node_type")
        if node_type == "array":
            problems.extend(_prefixed(entry_loc, array_problems_v3(node, context)))
        elif node_type == "group":
            problems.extend(_prefixed(entry_loc, group_problems_v3(node, context)))
    return tuple(problems)


__all__ = [
    "ArrayDocumentV3",
    "array_problems_v3",
    "consolidated_entries_problems",
    "group_problems_v3",
    "read_array_v3",
]
