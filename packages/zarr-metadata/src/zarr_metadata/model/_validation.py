"""Validation for Zarr metadata documents.

Validators check a document's JSON structure -- key presence, value
shapes, fixed literals like `zarr_format` -- and, in a v3 document, read
each extension point through the definition that claims its name in a
scope, so a configuration its definition refuses is refused here too. A
name nothing in the scope claims is left unjudged. A v3 fill value is
judged against the data type it names, the chunk grid against the
shape, and the codecs as a pipeline, each against the chunk it is
handed. Each concept gets a `validate_*` function returning every
problem found, an `is_*` type guard, and a `parse_*` function that
narrows or raises `MetadataValidationError`; a v3 array document also
gets `read_array_metadata_v3`, one read that returns what it read, the
problems, and the model when there are none.
The guards are `TypeGuard`s, not `TypeIs`: True narrows a value to its
document type, and False says nothing about its type, since a value can
be well typed and still not a valid document.

Every `ValidationProblem` carries a machine-readable `kind` alongside its
human-readable `message`, so consumers can dispatch on the failure mode
(`missing_key`, `invalid_type`, `invalid_value`, `invalid_json`,
`unknown_key`) without string-matching messages.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, TypeGuard, TypeVar, cast, get_args, get_origin

from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    nested_past_the_levels,
    not_an_object,
    outside_of,
    refine_json,
    refine_user_data,
    shown_key,
    with_input,
    within,
)
from zarr_metadata._json import is_canonical_json as _is_canonical_json
from zarr_metadata._sentinel import UNSET
from zarr_metadata._typed_json import typeddict_keys
from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON
from zarr_metadata.v2.definition import (
    CORE_V2,
    ZarrV2CodecDefinition,
    ZarrV2DataTypeDefinition,
    resolve_codec_v2,
    resolve_dtype_v2,
)
from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON
from zarr_metadata.v3._definition import (
    Chunk,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    DataTypeDefinition,
    Definition,
    Lengths,
    Resolved,
    StorageTransformerDefinition,
    chunk_grid_lengths,
    field_kind,
    fields_of,
    fill_value_problems,
    resolve,
)
from zarr_metadata.v3._pipeline import Stage, read_pipeline
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata._common import JSONValue
    from zarr_metadata._typed_json import Loc
    from zarr_metadata.model._array import ZarrV2ArrayMetadata, ZarrV3ArrayMetadata
    from zarr_metadata.v2.array import ZarrV2ArrayDimensionSeparator, ZarrV2ArrayOrder

# The standard top-level keys of a v3 array metadata document. Anything outside
# this set is an extension field. Built from the TypedDict's required/optional
# key sets (which resolve inherited keys, unlike `__annotations__`).
ARRAY_METADATA_REQUIRED_KEYS_V3: Final[frozenset[str]] = frozenset(
    ZarrV3ArrayMetadataJSON.__required_keys__
)
ARRAY_METADATA_OPTIONAL_KEYS_V3: Final[frozenset[str]] = frozenset(
    ZarrV3ArrayMetadataJSON.__optional_keys__
)
ARRAY_METADATA_STANDARD_KEYS_V3: Final[frozenset[str]] = (
    ARRAY_METADATA_REQUIRED_KEYS_V3 | ARRAY_METADATA_OPTIONAL_KEYS_V3
)

ARRAY_METADATA_REQUIRED_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2ArrayMetadataJSON.__required_keys__
)
ARRAY_METADATA_OPTIONAL_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2ArrayMetadataJSON.__optional_keys__
)
ARRAY_METADATA_STANDARD_KEYS_V2: Final[frozenset[str]] = (
    ARRAY_METADATA_REQUIRED_KEYS_V2 | ARRAY_METADATA_OPTIONAL_KEYS_V2
)

# The standard top-level keys of a v3 group metadata document. Anything outside
# this set is an extension field.
GROUP_METADATA_REQUIRED_KEYS_V3: Final[frozenset[str]] = frozenset(
    ZarrV3GroupMetadataJSON.__required_keys__
)
GROUP_METADATA_OPTIONAL_KEYS_V3: Final[frozenset[str]] = frozenset(
    ZarrV3GroupMetadataJSON.__optional_keys__
)
GROUP_METADATA_STANDARD_KEYS_V3: Final[frozenset[str]] = (
    GROUP_METADATA_REQUIRED_KEYS_V3 | GROUP_METADATA_OPTIONAL_KEYS_V3
)

GROUP_METADATA_REQUIRED_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2GroupMetadataJSON.__required_keys__
)
GROUP_METADATA_OPTIONAL_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2GroupMetadataJSON.__optional_keys__
)
GROUP_METADATA_STANDARD_KEYS_V2: Final[frozenset[str]] = (
    GROUP_METADATA_REQUIRED_KEYS_V2 | GROUP_METADATA_OPTIONAL_KEYS_V2
)


def missing_keys(
    required: frozenset[str], doc: Mapping[object, object]
) -> tuple[ValidationProblem, ...]:
    """One `missing_key` problem per required key absent from `doc`."""
    return tuple(
        ValidationProblem((key,), "missing required key", "missing_key")
        for key in sorted(required - doc.keys())
    )


def unexpected_keys(
    allowed: frozenset[str], doc: Mapping[object, object]
) -> tuple[ValidationProblem, ...]:
    """One problem per member outside a closed document's declared shape: `unknown_key`, as a closed TypedDict's checker reports one, so a caller who tolerates a member another writer added can tell it from a wrong value."""
    problems: list[ValidationProblem] = []
    for key in doc:
        if not isinstance(key, str):
            problems.append(
                ValidationProblem((), f"non-string document key {shown_key(key)}", "invalid_type")
            )
        elif key not in allowed:
            problems.append(ValidationProblem((key,), f"unexpected key {key!r}", "unknown_key"))
    return tuple(problems)


def check_literal(
    doc: Mapping[object, object], key: str, expected: object
) -> tuple[ValidationProblem, ...]:
    """One problem if `doc[key]` is present but not `expected`: of its type, when it is not of `expected`'s JSON type, else of its value."""
    if key in doc and (type(doc[key]) is not type(expected) or doc[key] != expected):
        return (outside_of((key,), doc[key], (expected,)),)
    return ()


def other_members_problems(
    doc: Mapping[object, object],
    standard_keys: frozenset[str],
    *,
    additional_reserved_keys: frozenset[str] = frozenset(),
) -> tuple[ValidationProblem, ...]:
    """Every key a string, and every member outside `standard_keys` a JSON value.

    For a document open to other members: v3 extension fields, and the
    members a v2 array's readers ignore.
    """
    return other_members(doc, standard_keys, additional_reserved_keys=additional_reserved_keys)[1]


def other_members(
    doc: Mapping[object, object],
    standard_keys: frozenset[str],
    *,
    additional_reserved_keys: frozenset[str] = frozenset(),
    at: Loc = (),
) -> tuple[dict[str, JSONValue], tuple[ValidationProblem, ...]]:
    """Each member outside `standard_keys` refined to JSON, and every problem, as `other_members_problems` finds them; a member that is not JSON is left out."""
    members: dict[str, JSONValue] = {}
    problems: list[ValidationProblem] = []
    reserved_keys = standard_keys | additional_reserved_keys
    for key, value in doc.items():
        if not isinstance(key, str):
            problems.append(
                ValidationProblem((), f"non-string top-level key {shown_key(key)}", "invalid_type")
            )
            continue
        if key in reserved_keys:
            continue
        refined, found = refine_json(value, (*at, key))
        problems.extend(within(found, at))
        if len(found) == 0:
            members[key] = refined
    return members, tuple(problems)


def _is_array(value: object) -> TypeGuard[Sequence[object]]:
    """Whether `value` reads as a JSON array: a sequence that is not a string or bytes.

    `str`, `bytes` and `bytearray` are sequences to Python, and none of them
    is an array to JSON. A `TypeGuard`, not a `TypeIs`: a `str` is a
    `Sequence[object]` this says no to.
    """
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_int_sequence(value: object) -> TypeGuard[Sequence[int]]:
    """Whether `value` is a JSON array of integers.

    JSON booleans decode to `bool`, which is an `int` subclass in Python but
    is not an integer in a metadata document, so booleans are excluded. A
    `TypeGuard`, not a `TypeIs`: `bytes` is a `Sequence[int]` this says no to.
    """
    return _is_array(value) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def dimension_lengths(
    doc: Mapping[object, object], key: str
) -> tuple[tuple[int, ...] | None, tuple[ValidationProblem, ...]]:
    """The dimension lengths `doc` holds at `key` (`shape`, `chunks`), and every problem with them.

    Dimension lengths are non-negative integers; the lengths are None when
    `doc` holds none at `key`, or ones with a problem.
    """
    if key not in doc:
        return None, ()
    value = doc[key]
    if not _is_int_sequence(value):
        return None, (ValidationProblem((key,), "expected an array of integers", "invalid_type"),)
    if any(item < 0 for item in value):
        return None, (ValidationProblem((key,), "expected non-negative integers", "invalid_value"),)
    return tuple(value), ()


def _is_canonical_dtype_v2(value: object) -> bool:
    """Whether a validated v2 dtype uses the tuple-backed public representation."""
    if isinstance(value, str):
        return True
    if not isinstance(value, tuple):
        return False
    for record in cast("tuple[object, ...]", value):
        if not isinstance(record, tuple):
            return False
        fields = cast("tuple[object, ...]", record)
        if not _is_canonical_dtype_v2(fields[1]):
            return False
        if len(fields) == 3 and not isinstance(fields[2], tuple):
            return False
    return True


def _is_canonical_metadata_field_v3(value: object) -> bool:
    """Whether a validated v3 metadata field has its declared runtime container type."""
    return isinstance(value, (str, dict))


def _is_canonical_array_metadata_v3(value: object) -> bool:
    """Whether a validated v3 array document matches `ZarrV3ArrayMetadataJSON` at runtime."""
    if not isinstance(value, dict):
        return False
    doc = cast("dict[str, object]", value)
    if not isinstance(doc["shape"], tuple) or not isinstance(doc["codecs"], tuple):
        return False
    if "storage_transformers" in doc and not isinstance(doc["storage_transformers"], tuple):
        return False
    if "dimension_names" in doc and not isinstance(doc["dimension_names"], tuple):
        return False
    if not all(_is_canonical_metadata_field_v3(doc[key]) for key, _ in _EXTENSION_POINTS_V3):
        return False
    if not all(
        _is_canonical_metadata_field_v3(item) for item in cast("tuple[object, ...]", doc["codecs"])
    ):
        return False
    return "storage_transformers" not in doc or all(
        _is_canonical_metadata_field_v3(item)
        for item in cast("tuple[object, ...]", doc["storage_transformers"])
    )


def _is_canonical_array_metadata_v2(value: object) -> bool:
    """Whether a validated v2 array document matches `ZarrV2ArrayMetadataJSON` at runtime."""
    if not isinstance(value, dict):
        return False
    doc = cast("dict[str, object]", value)
    if not isinstance(doc["shape"], tuple) or not isinstance(doc["chunks"], tuple):
        return False
    if not _is_canonical_dtype_v2(doc["dtype"]):
        return False
    compressor = doc["compressor"]
    if compressor is not None and not isinstance(compressor, dict):
        return False
    filters = doc["filters"]
    return filters is None or (
        isinstance(filters, tuple)
        and all(isinstance(item, dict) for item in cast("tuple[object, ...]", filters))
    )


def validate_attributes(value: object) -> tuple[ValidationProblem, ...]:
    """Validate an `attributes` value: a mapping with string keys.

    Returns a problem at `("attributes",)` if it is not, else `[]`. Shared by the
    v2 and v3 validators. Unlike the other `validate_*` functions (which
    return value-relative locs for the caller to `_prefix`), this emits the
    already-parent-relative `("attributes",)` loc, since it is only ever called
    with a document's `attributes` value.
    """
    return attributes_of(value)[1]


def members_past_the_levels(
    doc: Mapping[object, object], at: Loc
) -> dict[object, ValidationProblem]:
    """Each member of `doc`, a document that sits at `at`, that is a container past the levels a reader walks, with the problem it is, located in the document.

    A reader reports them and walks no further into them, as `refine_json`
    of the whole document would judge them: a document a level before the
    cap holds its scalars, and nothing else.
    """
    past: dict[object, ValidationProblem] = {}
    for key, item in doc.items():
        if isinstance(key, str):
            problem = nested_past_the_levels(item, (*at, key))
            if problem is not None:
                (past[key],) = within((problem,), at)
    return past


def attributes_of(
    value: object, at: Loc = ()
) -> tuple[dict[str, JSONValue] | None, tuple[ValidationProblem, ...]]:
    """An `attributes` value refined as user data, and every problem `validate_attributes` finds; None when it is not an object with string keys, or holds a value that is not JSON.

    `at` is where the document holding it sits in the one handed in, so
    the levels a reader walks are counted from that one's root; the
    problems are located in the document holding it.
    """
    if not isinstance(value, Mapping) or not all(
        isinstance(k, str) for k in cast("Mapping[object, object]", value)
    ):
        return None, (
            ValidationProblem(
                ("attributes",), "expected an object with string keys", "invalid_type"
            ),
        )
    attributes: dict[str, JSONValue] = {}
    problems: list[ValidationProblem] = []
    for key, item in cast("Mapping[str, object]", value).items():
        refined, found = refine_user_data(item, (*at, "attributes", key))
        problems.extend(found)
        attributes[key] = refined
    return (attributes if len(problems) == 0 else None), within(problems, at)


_MEMBERS_V3: Final = typeddict_keys(ZarrV3ArrayMetadataJSON).members

_EXTENSION_POINTS_V3: Final[tuple[tuple[str, type[Definition[Any]]], ...]] = tuple(
    (key, kind)
    for key, (annotation, _) in _MEMBERS_V3.items()
    if (kind := field_kind(annotation)) is not None
)
"""A v3 array document's single extension points, and the kind each is read as: each member its TypedDict annotates with a field alias."""

_EXTENSION_LISTS_V3: Final[tuple[tuple[str, type[Definition[Any]]], ...]] = tuple(
    (key, kind)
    for key, (annotation, _) in _MEMBERS_V3.items()
    if get_origin(annotation) is tuple and (kind := field_kind(get_args(annotation)[0])) is not None
)
"""Its lists of extension points, and the kind each entry is read as: each member it annotates as a tuple of a field alias."""


@dataclass(frozen=True, slots=True)
class ZarrV3ArrayMetadataReading:
    """A v3 array document as a scope read it, whatever it holds: each extension point, its codecs as a pipeline, every problem, and the model when there is none.

    A field the document does not hold is `UNSET`, and a list of them it does
    not hold as a list is empty.
    """

    data_type: Resolved[DataTypeDefinition[Any]] | UNSET = UNSET
    """The data type, as the scope read it."""
    chunk_grid: Resolved[ChunkGridDefinition[Any]] | UNSET = UNSET
    """The chunk grid, as the scope read it."""
    chunk_key_encoding: Resolved[ChunkKeyEncodingDefinition[Any]] | UNSET = UNSET
    """The chunk key encoding, as the scope read it."""
    chunk: Chunk = dataclasses.field(default_factory=Chunk)
    """The chunks the codecs are handed: the lengths the grid's chunks take along each axis of the shape, of the data type."""
    pipeline: tuple[Stage, ...] = ()
    """The codecs, read as a pipeline: each as the scope read it, with the chunk it is handed."""
    storage_transformers: tuple[Resolved[StorageTransformerDefinition[Any]], ...] = ()
    """The storage transformers, each as the scope read it."""
    problems: tuple[ValidationProblem, ...] = ()
    """Every reason the document is not a valid one."""
    metadata: ZarrV3ArrayMetadata | None = None
    """The document's model, holding these fields, when there is no problem; None otherwise."""

    def __reduce__(self) -> str | tuple[object, ...]:
        # A reading that holds its model pickles and copies as the model
        # does -- the pair, read once on load -- and comes back as that
        # model's own reading, so one model per document still.
        if self.metadata is not None:
            return (reading_of, (self.metadata,))
        return object.__reduce__(self)

    def fields(self) -> Iterator[tuple[Loc, Resolved[Any]]]:
        """Each field the document holds, as the scope read it, with where it sits in the document.

        The extension points, then each codec and storage transformer at its
        index, each followed by the fields it holds, as `fields_of` gives
        them: a shard's codecs, a struct's field types. `with_problems`
        gives each with its problems.
        """
        for key, field in (
            ("data_type", self.data_type),
            ("chunk_grid", self.chunk_grid),
            ("chunk_key_encoding", self.chunk_key_encoding),
        ):
            if field is not UNSET:
                yield from fields_of(cast("Resolved[Any]", field), (key,))
        for index, stage in enumerate(self.pipeline):
            yield from fields_of(stage.codec, ("codecs", index))
        for index, transformer in enumerate(self.storage_transformers):
            yield from fields_of(transformer, ("storage_transformers", index))


M = TypeVar("M")


def reading_of(model: object) -> object:
    """The reading `model`, a v3 model, holds: what a pickled reading that held a model is built again as."""
    return cast("Any", model).reading


def construct(model: type[M], /, **members: object) -> M:
    """A `model` of `members` a read found nothing wrong with: as its constructor builds one, without checking them again.

    What pydantic's `model_construct` is to its `__init__`. A model checks
    itself when it is built; a read has checked what it read already, so
    the model it builds is not checked twice. A member not given, or one
    the constructor does not take, is its default.
    """
    built = object.__new__(model)
    declared = dataclasses.fields(cast("Any", model))
    unknown = members.keys() - {member.name for member in declared if member.init}
    if len(unknown) != 0:
        msg = f"{model.__name__} has no member {sorted(unknown)!r} to build"
        raise TypeError(msg)
    for member in declared:
        if member.init and member.name in members:
            value = members[member.name]
        elif member.default is not dataclasses.MISSING:
            value = member.default
        elif member.default_factory is not dataclasses.MISSING:
            value = member.default_factory()
        else:
            msg = f"{model.__name__} is built with {member.name!r}"
            raise TypeError(msg)
        object.__setattr__(built, member.name, value)
    return built


@dataclass(frozen=True, slots=True)
class ArrayMembersV3:
    """The members of a v3 array document a read found nothing wrong with, other than its fields, refined as the model holds them."""

    shape: tuple[int, ...]
    fill_value: JSONValue
    dimension_names: tuple[str | None, ...] | UNSET
    attributes: dict[str, JSONValue]
    extra_fields: dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class ArrayMembersV2:
    """The members of a v2 array document a read found nothing wrong with, other than its fields, refined as the model holds them."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    fill_value: JSONValue
    order: ZarrV2ArrayOrder
    dimension_separator: ZarrV2ArrayDimensionSeparator
    attributes: dict[str, JSONValue] | UNSET
    extra_fields: dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class ZarrV2ArrayMetadataReading:
    """A v2 array document as a scope read it, whatever it holds: its dtype, compressor and filters, every problem, and the model when there is none.

    A field the document does not hold is `UNSET`; a `compressor` or
    `filters` written as `null` is None.
    """

    dtype: Resolved[ZarrV2DataTypeDefinition[Any]] | UNSET = UNSET
    """The dtype, as the scope read it."""
    compressor: Resolved[ZarrV2CodecDefinition[Any]] | UNSET | None = UNSET
    """The compressor, as the scope read it; None when written as `null`."""
    filters: tuple[Resolved[ZarrV2CodecDefinition[Any]], ...] | UNSET | None = UNSET
    """The filters, each as the scope read it; None when written as `null`."""
    problems: tuple[ValidationProblem, ...] = ()
    """Every reason the document is not a valid one."""
    metadata: ZarrV2ArrayMetadata | None = None
    """The document's model, holding these fields, when there is no problem; None otherwise."""

    def __reduce__(self) -> str | tuple[object, ...]:
        # As the v3 reading: through its model, when it holds one.
        if self.metadata is not None:
            return (reading_of, (self.metadata,))
        return object.__reduce__(self)

    def fields(self) -> Iterator[tuple[Loc, Resolved[Any]]]:
        """Each field the document holds, as the scope read it, where it sits: the dtype, a struct's record types after it, the compressor, each filter at its index."""
        if self.dtype is not UNSET:
            yield from fields_of(cast("Resolved[Any]", self.dtype), ("dtype",))
        if self.compressor is not UNSET and self.compressor is not None:
            yield from fields_of(self.compressor, ("compressor",))
        if self.filters is not UNSET and self.filters is not None:
            for index, entry in enumerate(self.filters):
                yield from fields_of(entry, ("filters", index))


def read_array_v3(
    value: object, context: Context, *, at: Loc = ()
) -> tuple[ZarrV3ArrayMetadataReading, ArrayMembersV3 | None]:
    """`value`, a v3 array document, as `context` read it, without its model, and its other members refined; None when it has a problem.

    `read_array_metadata_v3` builds the model from the two. A field object
    in the document -- a `Read` built by hand -- is not JSON, and is refused
    as such. `at` is where the document sits in the one handed in -- a document consolidated metadata holds sits three levels below
    its group's -- so the levels a reader walks are counted from that
    one's root; the problems are located in this document.
    """
    if not isinstance(value, Mapping):
        return ZarrV3ArrayMetadataReading(problems=not_an_object(value)), None
    doc = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = list(missing_keys(ARRAY_METADATA_REQUIRED_KEYS_V3, doc))
    past = members_past_the_levels(doc, at)
    problems.extend(past.values())
    whole, doc = doc, {key: item for key, item in doc.items() if key not in past}
    extra_fields, found = other_members(doc, ARRAY_METADATA_STANDARD_KEYS_V3, at=at)
    problems.extend(found)
    problems.extend(check_literal(doc, "zarr_format", 3))
    problems.extend(check_literal(doc, "node_type", "array"))
    shape, shape_problems = dimension_lengths(doc, "shape")
    problems.extend(shape_problems)
    # Each extension point is read by `resolve`, which judges its envelope
    # -- every extension *point* must be understood, so a `must_understand`
    # of `false` is refused at each: ignoring a codec gives wrong bytes as
    # surely as ignoring a data type gives wrong values, and the spec
    # naming only the first three
    # (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581)
    # is read as an oversight rather than a licence -- and then its
    # configuration, against the definition in `context` that claims its
    # name. `must_understand: false` keeps its meaning where it has one: an
    # unknown top-level extension *field*, which a reader really can skip.
    read: dict[str, Resolved[Any]] = {}
    for key, kind in _EXTENSION_POINTS_V3:
        if key in doc:
            read[key], found = resolve(doc[key], kind, context, (*at, key))
            problems.extend(within(found, at))
    # The fill value is JSON, and judged by the data type the scope read,
    # when there is one: a data type nothing in scope claims leaves it
    # unjudged.
    fill_value: JSONValue = None
    if "fill_value" in doc:
        fill_value, found = refine_json(doc["fill_value"], (*at, "fill_value"))
        problems.extend(within(found, at))
        if len(found) == 0 and "data_type" in read:
            problems.extend(fill_value_problems(read["data_type"], fill_value, ("fill_value",)))
    # The chunk grid is judged against the shape, once both are read, and
    # says the lengths of the chunks the first codec is handed: an entry
    # for each dimension of the shape, None where nothing says it.
    lengths: Lengths | None = None if shape is None else (None,) * len(shape)
    if "chunk_grid" in read and shape is not None:
        lengths, found = chunk_grid_lengths(read["chunk_grid"], shape, ("chunk_grid",))
        problems.extend(found)
    listed: dict[str, list[Resolved[Any]]] = {}
    for key, kind in _EXTENSION_LISTS_V3:
        if key in doc:
            entries = doc[key]
            if not _is_array(entries):
                problems.append(ValidationProblem((key,), "expected an array", "invalid_type"))
            else:
                listed[key] = []
                for index, entry in enumerate(entries):
                    resolved, found = resolve(entry, kind, context, (*at, key, index))
                    listed[key].append(resolved)
                    problems.extend(within(found, at))
    # The codecs are read as a pipeline, the first handed the grid's chunks
    # of the array's data type: in order, each judged against the chunk it
    # is handed. That holds one array -> bytes codec, so it is not empty.
    chunk = Chunk(lengths, read.get("data_type"))
    pipeline: tuple[Stage, ...] = ()
    if "codecs" in listed:
        pipeline, found = read_pipeline(listed["codecs"], chunk, ("codecs",))
        problems.extend(found)
    attributes: dict[str, JSONValue] | None = {}
    if "attributes" in doc:
        attributes, found = attributes_of(doc["attributes"], at)
        problems.extend(found)
    if "dimension_names" in doc:
        # Simple typed sequences (dimension_names, shape, chunks) report a single
        # field-level loc, not per-bad-item locs; per-index locs are reserved for
        # the metadata-field lists (codecs, storage_transformers).
        names = doc["dimension_names"]
        if not _is_array(names):
            problems.append(
                ValidationProblem(("dimension_names",), "expected an array", "invalid_type")
            )
        elif not all(item is None or isinstance(item, str) for item in names):
            problems.append(
                ValidationProblem(
                    ("dimension_names",), "expected an array of strings or null", "invalid_type"
                )
            )
        elif shape is not None and len(names) != len(shape):
            problems.append(
                ValidationProblem(
                    ("dimension_names",),
                    "expected one name per dimension of shape",
                    "invalid_value",
                )
            )
    reading = ZarrV3ArrayMetadataReading(
        data_type=read.get("data_type", UNSET),
        chunk_grid=read.get("chunk_grid", UNSET),
        chunk_key_encoding=read.get("chunk_key_encoding", UNSET),
        chunk=chunk,
        pipeline=pipeline,
        storage_transformers=tuple(listed.get("storage_transformers", ())),
        problems=with_input(problems, whole),
    )
    if len(problems) != 0 or shape is None or attributes is None:
        return reading, None
    names = doc.get("dimension_names", UNSET)
    members = ArrayMembersV3(
        shape=shape,
        fill_value=fill_value,
        dimension_names=UNSET if names is UNSET else tuple(cast("Sequence[str | None]", names)),
        attributes=attributes,
        extra_fields=extra_fields,
    )
    return reading, members


def validate_array_metadata_v3(
    value: object, *, context: Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a valid v3 array document.

    Its structure, and each extension point read through the definition
    that claims its name in `context`: a gzip `level` out of range, a key a
    codec's configuration does not declare. The fill value is judged
    against the data type as `context` read it -- an `int8` fill value of
    300 -- and the chunk grid against the shape: a regular grid with a
    chunk length for each of two dimensions, over an array of three. The
    codecs are read as a pipeline: in order, each judged against the chunk
    it is handed -- a `transpose` whose `order` has another number of
    axes, a shard its inner chunks do not divide -- and a shard's inner
    and index codecs too.
    A name nothing in `context` claims is left unjudged, with any fill
    value of it, and a codec of that name leaves the codec after it
    handed a chunk nothing is known of. Unknown top-level keys are
    allowed (they map to `extra_fields`); a reader must understand each
    one that does not say `must_understand: false`, which the model
    reports as `must_understand_fields`. These are the `problems` of
    `read_array_metadata_v3`, which holds what was read to find them.
    """
    scope = CORE_AND_EXTENSIONS if context is None else context
    return read_array_v3(value, scope)[0].problems


def is_array_metadata_v3(
    value: object, *, context: Context | None = None
) -> TypeGuard[ZarrV3ArrayMetadataJSON]:
    """Whether `value` is a v3 array document `validate_array_metadata_v3` finds nothing wrong with, written with tuples."""
    scope = CORE_AND_EXTENSIONS if context is None else context
    return (
        _is_canonical_json(value, finite=False)
        and not validate_array_metadata_v3(value, context=scope)
        and _is_canonical_array_metadata_v3(value)
    )


def parse_array_metadata_v3(
    value: object, *, context: Context | None = None
) -> ZarrV3ArrayMetadataJSON:
    """Return `value` as `ZarrV3ArrayMetadataJSON`, or raise `MetadataValidationError`."""
    scope = CORE_AND_EXTENSIONS if context is None else context
    problems = validate_array_metadata_v3(value, context=scope)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3ArrayMetadataJSON", arrays_to_tuples(value))


def read_array_v2(
    value: object, context: Context, *, at: Loc = ()
) -> tuple[ZarrV2ArrayMetadataReading, ArrayMembersV2 | None]:
    """`value`, a v2 array document, read once in `context`: the reading, and its other members when nothing is wrong.

    `dtype`, `compressor` and `filters` are read in `context`: a dtype or
    codec the scope refuses is a problem, one it does not claim is not;
    `fill_value` is judged by the dtype the scope read. `compressor` and
    `filters` are required keys that may be `None`. `at` is where the
    document sits in the one handed in, which prefixes every problem.
    """
    if not isinstance(value, Mapping):
        return ZarrV2ArrayMetadataReading(problems=within(not_an_object(value), at)), None
    doc = cast("Mapping[object, object]", value)
    # Unlike the group document ("Other keys MUST NOT be present",
    # https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L313), the v2 array document is open: other keys "SHOULD NOT be
    # present within the metadata object and SHOULD be ignored by
    # implementations" (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L91-L92), so a member outside
    # ARRAY_METADATA_STANDARD_KEYS_V2 is not a problem for being there. Ignored
    # is not unchecked: it is JSON, and its key a string, as in v3.
    problems: list[ValidationProblem] = list(missing_keys(ARRAY_METADATA_REQUIRED_KEYS_V2, doc))
    extra_fields, found = other_members(doc, ARRAY_METADATA_STANDARD_KEYS_V2)
    problems.extend(found)
    problems.extend(check_literal(doc, "zarr_format", 2))
    shape, shape_problems = dimension_lengths(doc, "shape")
    chunks, chunks_problems = dimension_lengths(doc, "chunks")
    problems.extend(shape_problems)
    problems.extend(chunks_problems)
    if shape is not None and chunks is not None and len(shape) != len(chunks):
        problems.append(
            ValidationProblem(
                ("chunks",),
                "expected the same number of dimensions as shape",
                "invalid_value",
            )
        )
    dtype: Resolved[ZarrV2DataTypeDefinition[Any]] | UNSET = UNSET
    if "dtype" in doc:
        # A typestr by its family, field records as a struct.
        dtype, found = resolve_dtype_v2(doc["dtype"], context, ("dtype",))
        problems.extend(found)
    if "order" in doc and doc["order"] not in ("C", "F"):
        problems.append(outside_of(("order",), doc["order"], ("C", "F")))
    compressor: Resolved[ZarrV2CodecDefinition[Any]] | UNSET | None = UNSET
    if "compressor" in doc:
        compressor = None
        if doc["compressor"] is not None:
            compressor, found = resolve_codec_v2(doc["compressor"], context, ("compressor",))
            problems.extend(found)
    filters: tuple[Resolved[ZarrV2CodecDefinition[Any]], ...] | UNSET | None = UNSET
    if "filters" in doc:
        filters = None
        entries = doc["filters"]
        if entries is not None:
            if not _is_array(entries):
                problems.append(
                    ValidationProblem(
                        ("filters",),
                        "expected null or an array of codec configurations",
                        "invalid_type",
                    )
                )
            else:
                # "A list of JSON objects providing codec configurations, or
                # null" (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L76-L79): an empty list is a list.
                read: list[Resolved[ZarrV2CodecDefinition[Any]]] = []
                for index, item in enumerate(entries):
                    entry, found = resolve_codec_v2(item, context, ("filters", index))
                    read.append(entry)
                    problems.extend(found)
                filters = tuple(read)
    if "dimension_separator" in doc and doc["dimension_separator"] not in (".", "/"):
        problems.append(
            outside_of(("dimension_separator",), doc["dimension_separator"], (".", "/"))
        )
    fill_value: JSONValue = None
    if "fill_value" in doc:
        # JSON, judged by the dtype the scope read, when there is one: a
        # dtype nothing in scope claims leaves it unjudged.
        fill_value, found = refine_json(doc["fill_value"], ("fill_value",))
        problems.extend(found)
        if len(found) == 0 and dtype is not UNSET:
            problems.extend(fill_value_problems(dtype, fill_value, ("fill_value",)))
    attributes: dict[str, JSONValue] | UNSET | None = UNSET
    if "attributes" in doc:
        attributes, found = attributes_of(doc["attributes"])
        problems.extend(found)
    reading = ZarrV2ArrayMetadataReading(
        dtype=dtype,
        compressor=compressor,
        filters=filters,
        problems=within(with_input(problems, doc), at),
    )
    if len(problems) != 0 or shape is None or chunks is None or attributes is None:
        return reading, None
    members = ArrayMembersV2(
        shape=shape,
        chunks=chunks,
        fill_value=fill_value,
        order=cast("ZarrV2ArrayOrder", doc["order"]),
        dimension_separator=cast(
            "ZarrV2ArrayDimensionSeparator", doc.get("dimension_separator", ".")
        ),
        attributes=attributes,
        extra_fields=extra_fields,
    )
    return reading, members


def validate_array_metadata_v2(
    value: object, *, context: Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Every reason `value` is not a valid v2 array document, read in `context`, `CORE_V2` when none is given.

    `dtype`, `compressor` and `filters` are read in the scope: a dtype or
    codec the scope refuses is a problem, one it does not claim is not;
    `fill_value` is judged by the dtype the scope read.
    """
    return read_array_v2(value, CORE_V2 if context is None else context)[0].problems


def is_array_metadata_v2(
    value: object, *, context: Context | None = None
) -> TypeGuard[ZarrV2ArrayMetadataJSON]:
    """Whether `value` is a valid v2 array metadata document, read in `context`, `CORE_V2` when none is given."""
    return (
        _is_canonical_json(value, finite=False)
        and not validate_array_metadata_v2(value, context=context)
        and _is_canonical_array_metadata_v2(value)
    )


def parse_array_metadata_v2(
    value: object, *, context: Context | None = None
) -> ZarrV2ArrayMetadataJSON:
    """`value` as `ZarrV2ArrayMetadataJSON`, read in `context`, `CORE_V2` when none is given; `MetadataValidationError` with every problem."""
    problems = validate_array_metadata_v2(value, context=context)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV2ArrayMetadataJSON", arrays_to_tuples(value))


def validate_group_metadata_v2(
    value: object, *, context: Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a structurally-valid v2 group doc.

    Validates the in-memory merged form: the `.zgroup` fields plus an
    optional `attributes` mapping folded in from `.zattrs`. A group holds
    no field a scope reads; `context` is taken as every v2 reader takes it.
    """
    if not isinstance(value, Mapping):
        return not_an_object(value)
    doc = cast("Mapping[object, object]", value)
    problems: list[ValidationProblem] = list(missing_keys(GROUP_METADATA_REQUIRED_KEYS_V2, doc))
    problems.extend(unexpected_keys(GROUP_METADATA_STANDARD_KEYS_V2, doc))
    problems.extend(check_literal(doc, "zarr_format", 2))
    if "attributes" in doc:
        problems.extend(validate_attributes(doc["attributes"]))
    return with_input(problems, doc)


def is_group_metadata_v2(
    value: object, *, context: Context | None = None
) -> TypeGuard[ZarrV2GroupMetadataJSON]:
    """Whether `value` is a structurally-valid v2 group metadata document; `context` is taken as every v2 reader takes it."""
    return _is_canonical_json(value, finite=False) and not validate_group_metadata_v2(
        value, context=context
    )


def parse_group_metadata_v2(
    value: object, *, context: Context | None = None
) -> ZarrV2GroupMetadataJSON:
    """`value` narrowed to `ZarrV2GroupMetadataJSON`, or `MetadataValidationError`; `context` is taken as every v2 reader takes it."""
    problems = validate_group_metadata_v2(value, context=context)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast(ZarrV2GroupMetadataJSON, arrays_to_tuples(value))


StoreKey = TypeVar("StoreKey", bound=str)
"""The key type of a mapping of store keys to bytes: `str`, or the literal keys one document names."""


def load_store_json(mapping: Mapping[StoreKey, bytes], key: str) -> object:
    """Decode the JSON document stored at `key` in `mapping`.

    Returns `object`, not `Any`: what a store holds is unknown until a
    validator says otherwise, and `Any` would let unchecked values flow
    into typed positions silently. Narrow the result with a `parse_*`.

    Decoding is Python's, so `NaN`, `Infinity` and `-Infinity` are read as
    the floats they spell, as zarr-python writes attributes; where one may
    be is the document's validator's to say. Every ingestion failure here
    surfaces as `MetadataValidationError`: a missing store key is a
    `missing_key` problem, a value that is not `bytes` an `invalid_type`
    problem, and undecodable bytes an `invalid_json` problem, rather than
    leaking `KeyError`, `TypeError` or `json.JSONDecodeError` to
    callers.
    """
    # Read by a `str` key whatever narrower key type the mapping declares:
    # a key it does not hold is only absent.
    stored = cast("Mapping[str, bytes]", mapping)
    if key not in stored:
        raise MetadataValidationError(
            [ValidationProblem((key,), "missing store key", "missing_key")]
        )
    # The runtime half of the annotation: `json.loads` decodes a `str` and
    # raises `TypeError` on most else.
    raw = cast("object", stored[key])
    if not isinstance(raw, bytes):
        refused = ValidationProblem(
            (key,), f"expected bytes, got {type(raw).__name__}", "invalid_type"
        )
        raise MetadataValidationError(with_input((refused,), stored))
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise MetadataValidationError(
            [ValidationProblem((key,), f"invalid JSON: {exc}", "invalid_json")]
        ) from exc


def dump_store_json(value: object, *, indent: int | str | None = None) -> bytes:
    """Encode a document its validator has passed as JSON bytes.

    A non-finite number is written as Python's `json` writes it (`NaN`,
    `Infinity`, `-Infinity`), as zarr-python writes attributes; the
    validator is what keeps one out of anywhere else.
    """
    return json.dumps(value, indent=indent, allow_nan=True).encode("utf-8")
