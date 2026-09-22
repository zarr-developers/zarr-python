"""A whole v3 array document, read in three layers, each with what it needs.

1. `well_formed_array_v3`: the value alone. JSON syntax and the
   document's shape -- arrays as tuples, string keys, floats finite
   outside the user's attributes, the keys a v3 array has and the shapes
   their values take, the envelope of each extension point.
2. `read_array_v3`: a scope. Each extension point's name related to a
   class in the `Context`, and the class handed the field: the
   configuration parsed against its record, the rules asked, nested
   entities read the same way.
3. `refine_array_v3`: the array. The fill value against the data type,
   the grid against the shape, and the codec pipeline walked with what
   reaches each codec, which is the resolved pipeline -- the array each
   codec is handed, and a shard's inner pipelines refined the same way.
   Validation is what the walk finds.

Each layer hands the next a typed value and its problems; the next reads
what it can and never repeats the work of the one before. Nothing here
knows what `blosc` or `int32` or `rectilinear` is. A new extension is a
class and a registry entry, and this module does not change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeVar, cast

from zarr_metadata.model._validation import (
    MetadataValidationError,
    ValidationProblem,
    refine_node_json,
)
from zarr_metadata.model._validation import (
    validate_array_metadata_v3 as validate_array_metadata_v3_structure,
)
from zarr_metadata.v3._chain import Pipeline, refine_pipeline
from zarr_metadata.v3._entity import (
    ChunkGridEntity,
    ChunkKeyEncodingEntity,
    CodecEntity,
    DataTypeEntity,
    MetadataEntity,
    Opaque,
    StorageTransformerEntity,
    held_problems,
    problem,
    read_field,
    within,
)
from zarr_metadata.v3._parts import ArrayParts, ChunkGrid
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata._common import JSONValue
    from zarr_metadata.v3._entity import Loc


_EntityT = TypeVar("_EntityT", bound=MetadataEntity)


@dataclass(frozen=True, slots=True)
class ArrayDocumentV3:
    """A v3 array document with its extension points read as entities: the second layer's value.

    A field that could not be read holds an `Opaque`, which carries the
    JSON the document wrote and says whether the name was out of scope --
    an extension this reader does not model, which is not an error -- or
    claimed and refused. Both are narrowable: every field is an exhaustive
    two-case union. `refine_array_v3` takes it on to the third layer.
    """

    document: Mapping[str, JSONValue]
    data_type: DataTypeEntity | Opaque
    chunk_grid: ChunkGridEntity | Opaque
    chunk_key_encoding: ChunkKeyEncodingEntity | Opaque
    codecs: tuple[CodecEntity | Opaque, ...]
    storage_transformers: tuple[StorageTransformerEntity | Opaque, ...]

    def __post_init__(self) -> None:
        """Refuse a field holding anything but an entity of its kind or an `Opaque`.

        `read_array_v3` builds a document that holds what it says by
        construction; this is the same guarantee for one built by hand,
        located at the field.
        """
        found = (
            *_as_object(self.document),
            *held_problems(self.data_type, DataTypeEntity, ("data_type",)),
            *held_problems(self.chunk_grid, ChunkGridEntity, ("chunk_grid",)),
            *held_problems(
                self.chunk_key_encoding, ChunkKeyEncodingEntity, ("chunk_key_encoding",)
            ),
            *(
                entry
                for index, codec in enumerate(self.codecs)
                for entry in held_problems(codec, CodecEntity, ("codecs", index))
            ),
            *(
                entry
                for index, transformer in enumerate(self.storage_transformers)
                for entry in held_problems(
                    transformer, StorageTransformerEntity, ("storage_transformers", index)
                )
            ),
        )
        if len(found) != 0:
            raise MetadataValidationError(found)

    def canonical(self) -> ArrayDocumentV3:
        """This document in the simplest form that means the same thing.

        Each entity in its own canonical form and in its own spelling of
        the envelope -- the bare name when nothing is configured, no
        `must_understand`, which means what absence means -- and the one
        rule that is the document's own: `dimension_names` of nothing
        but nulls says what omitting the field says. A *transformation*,
        asked for by `canonicalize_array_metadata_v3`; `to_json` does
        not apply it. What comes back is a document written that way, so
        writing it changes nothing further.
        """
        simplified = replace(
            self,
            data_type=self.data_type.canonical(),
            chunk_grid=self.chunk_grid.canonical(),
            chunk_key_encoding=self.chunk_key_encoding.canonical(),
            codecs=tuple(codec.canonical() for codec in self.codecs),
            storage_transformers=tuple(entry.canonical() for entry in self.storage_transformers),
        )
        document = {**self.document, **_rendered(simplified)}
        names = document.get("dimension_names")
        if isinstance(names, tuple) and all(
            entry is None for entry in cast("tuple[object, ...]", names)
        ):
            del document["dimension_names"]
        return replace(simplified, document=document)

    def to_json(self) -> dict[str, JSONValue]:
        """The document as it was written, with each entity's members as the entity has them.

        Faithful: a document read and written comes out as it went in,
        an entity's envelope included -- `{"name": "crc32c"}` stays an
        object, an empty `configuration` and a `must_understand` of
        `true` stay written -- because the document knows the spelling
        it read and puts it back around what the entity writes. What
        changed is what changes: a member replaced through
        `with_configuration` is written as the entity now has it, and an
        entity put in by hand is written as it writes itself. A field
        the document did not have is not invented, and one it wrote as
        something no entity could be read from stands as written. Ask
        `canonical` first for the simplest equivalent spelling.
        """
        return {
            **self.document,
            **{
                key: cast("JSONValue", _as_written(self.document[key], value))
                for key, value in _rendered(self).items()
            },
        }

    @classmethod
    def from_json(cls, value: object, *, context: Context = CORE_AND_EXTENSIONS) -> ArrayDocumentV3:
        """A v3 array document read into entities, or raise.

        The reader's front door, and the one entry point that fails fast:
        all three layers, and either every extension point is read and
        the whole composes, or a single `MetadataValidationError` carries
        every reason it does not -- structural and semantic together. Use
        `validate_array_metadata_v3` instead when you want the problems
        as data, and the layers themselves when you want to stop between
        them.

        A name this `context` does not model is *not* a failure. It comes
        back as an `Opaque` marked `out_of_scope`, because a document may
        legitimately use an extension this reader does not know, and
        refusing it would make openness unimplementable. What fails is
        metadata that is wrong, not metadata that is unfamiliar.
        """
        document, problems = well_formed_array_v3(value)
        if document is not None:
            # Read whatever the shape allowed, so a structural problem does
            # not hide the semantic ones behind it.
            array, found = read_array_v3(document, context)
            _, composed = refine_array_v3(array)
            problems = (*problems, *found, *composed)
            if len(problems) == 0:
                return array
        raise MetadataValidationError(problems)


@dataclass(frozen=True, slots=True)
class RefinedArrayV3:
    """A v3 array document refined against its own array: the third layer's value.

    `parts` is the array the codec pipeline is handed -- its chunks,
    under its grid, of its data type -- and `pipeline` is that pipeline
    resolved: at each position the codec and what reaches it, a shard's
    inner pipelines refined inside it. What a codec pipeline is built
    from, and what validating the composition finds on the way.
    """

    array: ArrayDocumentV3
    parts: ArrayParts
    pipeline: Pipeline


def well_formed_array_v3(
    value: object,
) -> tuple[Mapping[str, JSONValue] | None, tuple[ValidationProblem, ...]]:
    """The first layer: `value` as a refined v3 array document, with every structural problem.

    Needs nothing but the value. The JSON is refined -- arrays as
    tuples, string keys, floats finite except in the attributes, which
    are user data (`refine_node_json`) -- and the document's shape is
    judged by the model layer: the keys a v3 array has, the shapes their
    values take, the envelope of each extension point. What comes back
    is refined JSON that the next layer reads without normalizing or
    judging JSON-ness again, and the structural problems beside it, which
    do not stop the next layer from reading what it can. A value that is
    not JSON, or not an object, is None with the reasons: not JSON is
    the first verdict, and there is nothing to read.
    """
    refined, problems = refine_node_json(value)
    if refined is None:
        return None, problems
    if not isinstance(refined, Mapping):
        return None, problem((), f"expected a v3 array document as an object, got {refined!r}")
    document = cast("Mapping[str, JSONValue]", refined)
    return document, validate_array_metadata_v3_structure(document)


def read_array_v3(
    document: Mapping[str, JSONValue], context: Context
) -> tuple[ArrayDocumentV3, tuple[ValidationProblem, ...]]:
    """The second layer: `document`'s extension points, read in `context`.

    Needs a scope. The one place that knows which of a document's fields
    holds which kind of entity; each is handed to `read_field`, its
    envelope having been judged with the document. Type-space only: what
    comes back is well-typed by construction, and the problems are the
    reasons some of it is not an entity.
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


def refine_array_v3(array: ArrayDocumentV3) -> tuple[RefinedArrayV3, tuple[ValidationProblem, ...]]:
    """The third layer: `array` against its own array, and the pipeline resolved.

    Needs the array: the fill value is judged by the data type it fills,
    the grid against the shape it divides, the dimension names counted
    against it, and the codec pipeline walked from the parts the grid
    and data type make, each codec handed what reaches it. A pipeline
    the document did not write as an array was not read, and is not
    judged as an empty one.
    """
    parts = _parts(array)
    pipeline, composed = (
        refine_pipeline(array.codecs, parts, ("codecs",))
        if _listed(array.document, "codecs") is not None
        else (Pipeline(()), ())
    )
    problems = (
        *_fill_value_problems(array),
        *_grid_problems(array),
        *_dimension_names_problems(array),
        *composed,
    )
    return RefinedArrayV3(array, parts, pipeline), problems


def _parts(array: ArrayDocumentV3) -> ArrayParts:
    """The array the codec pipeline is handed."""
    shape = array.document.get("shape")
    # A grid out of scope still divides an array of some rank, and the
    # shape is what pins it -- which is enough to catch a shard whose
    # inner chunk has the wrong number of dimensions.
    grid = (
        array.chunk_grid.grid(shape)
        if isinstance(array.chunk_grid, ChunkGridEntity)
        else ChunkGrid.unreadable(shape)
    )
    return ArrayParts(
        grid, array.data_type if isinstance(array.data_type, DataTypeEntity) else None
    )


def _as_object(value: object) -> tuple[ValidationProblem, ...]:
    """Why `value` is not the document as an object, which is what the reader read it as."""
    if isinstance(value, Mapping):
        return ()
    return problem((), f"expected the document as an object, got {value!r}")


def _rendered(array: ArrayDocumentV3) -> dict[str, JSONValue]:
    """Each entity field the document has, as its entities write it; one nothing was read from is left out."""
    rendered: dict[str, JSONValue] = {}
    for key, entity in (
        ("data_type", array.data_type),
        ("chunk_grid", array.chunk_grid),
        ("chunk_key_encoding", array.chunk_key_encoding),
    ):
        if key in array.document:
            rendered[key] = entity.to_json()
    for key, entities in (
        ("codecs", array.codecs),
        ("storage_transformers", array.storage_transformers),
    ):
        if _listed(array.document, key) is not None:
            rendered[key] = tuple(entity.to_json() for entity in entities)
    return rendered


_ENVELOPE_KEYS = frozenset({"name", "configuration"})
"""What an entity writes around its members; anything else around a name is the document's."""


def _as_written(original: object, rendered: object) -> object:
    """`rendered`, an entity's JSON, in the spelling `original`, the document's JSON at the same place, used.

    The envelope's writer, the counterpart of `named_configuration`. An
    entity writes its members and its own spelling of the envelope,
    since it has no document to be faithful to; the document has, so
    the spellings that mean the same come back as they were written:
    the object around a bare name, a `configuration` of nothing, a
    `must_understand` of `true`. The two trees are walked together, so
    a codec inside a shard is dressed as one in the pipeline is. Where
    they disagree -- an entity put in or taken out by hand, a name
    changed -- the rendered value stands, members and all.
    """
    nothing: dict[str, object] = {}
    if isinstance(original, Mapping):
        before = cast("Mapping[str, object]", original)
        if isinstance(rendered, str):
            if before.get("name") == rendered:
                return {
                    key: nothing if key == "configuration" else value
                    for key, value in before.items()
                }
            return rendered
        if isinstance(rendered, Mapping):
            after = cast("Mapping[str, object]", rendered)
            written = {key: _as_written(before.get(key), value) for key, value in after.items()}
            if (
                after.keys() <= _ENVELOPE_KEYS
                and "name" in after
                and before.get("name") == after["name"]
            ):
                for key, value in before.items():
                    if key not in written:
                        written[key] = nothing if key == "configuration" else value
            return written
    if isinstance(original, (list, tuple)) and isinstance(rendered, (list, tuple)):
        before = cast("Sequence[object]", original)
        after = cast("Sequence[object]", rendered)
        if len(before) != len(after):
            return tuple(after)
        return tuple(_as_written(entry, value) for entry, value in zip(before, after, strict=True))
    return rendered


def _listed(document: Mapping[str, JSONValue], key: str) -> Sequence[JSONValue] | None:
    """What the document lists at `key`; None if it wrote no array there."""
    entries = document.get(key)
    return cast("Sequence[JSONValue]", entries) if isinstance(entries, (list, tuple)) else None


def _read_one(
    context: Context, kind: type[_EntityT], document: Mapping[str, JSONValue], key: str
) -> tuple[_EntityT | Opaque, tuple[ValidationProblem, ...]]:
    """The entity of `kind` the document names at `key`; an `Opaque` if it names none."""
    value = document.get(key)
    if value is None:
        return Opaque.create_unchecked(None, "invalid"), ()
    return read_field(value, kind, context, (key,))


def _read_each(
    context: Context, kind: type[_EntityT], document: Mapping[str, JSONValue], key: str
) -> tuple[tuple[_EntityT | Opaque, ...], tuple[ValidationProblem, ...]]:
    """The entities of `kind` the document lists at `key`, in order."""
    entries = _listed(document, key)
    if entries is None:
        return (), ()
    read: list[_EntityT | Opaque] = []
    problems: list[ValidationProblem] = []
    for index, entry in enumerate(entries):
        entity, found = read_field(entry, kind, context, (key, index))
        read.append(entity)
        problems.extend(found)
    return tuple(read), tuple(problems)


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
    document: Mapping[str, JSONValue], context: Context
) -> tuple[ValidationProblem, ...]:
    """The second and third layers' problems for a refined document, together.

    For a document the first layer has refined; a consolidated child is
    one.
    """
    array, problems = read_array_v3(document, context)
    _, composed = refine_array_v3(array)
    return (*problems, *composed)


def _prefixed(loc: Loc, problems: Sequence[ValidationProblem]) -> tuple[ValidationProblem, ...]:
    """Re-base every problem's `loc` under `loc`, for a nested document."""
    return tuple(
        ValidationProblem((*loc, *found.loc), found.message, found.kind) for found in problems
    )


def _as_string_mapping(value: object) -> Mapping[str, JSONValue] | None:
    """`value` as a string-keyed mapping, or None if it is not one."""
    if not isinstance(value, Mapping):
        return None
    mapping = cast("Mapping[object, object]", value)
    if any(not isinstance(key, str) for key in mapping):
        return None
    return cast("Mapping[str, JSONValue]", mapping)


def group_problems_v3(
    document: Mapping[str, JSONValue], context: Context = CORE_AND_EXTENSIONS
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
    "RefinedArrayV3",
    "array_problems_v3",
    "consolidated_entries_problems",
    "group_problems_v3",
    "read_array_v3",
    "refine_array_v3",
    "well_formed_array_v3",
]
