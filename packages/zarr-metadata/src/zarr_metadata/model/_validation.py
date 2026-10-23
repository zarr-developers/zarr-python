"""Validation for Zarr metadata documents.

Validators check a document's JSON structure: key presence, value
shapes, and fixed literals like `zarr_format`. In a v3 document they
also read each extension point through the definition that claims its
name in a scope, so a configuration that its definition refuses is
refused here too. A name that nothing in the scope claims is not
checked. A v3 fill value is checked against the data type the document
names, the chunk grid against the shape, and the codecs as a pipeline,
each codec against the chunk it receives. Each concept gets a
`validate_*` function that returns every problem found, an `is_*` type
guard, and a `parse_*` function that narrows the value or raises
`MetadataValidationError`. A v3 array document also gets
`read_array_metadata_v3`, which does one read and returns what it read,
the problems, and the model when there are no problems.
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
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    is_object,
    is_tuple,
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
from zarr_metadata.definition._context import (
    Context,
    ZarrV2Context,
    ZarrV3Context,
    scoped,
)
from zarr_metadata.definition._definition import (
    Definition,
    ResolvedField,
    field_kind,
    fields_of,
    fill_value_problems,
    resolve_refined,
)
from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON
from zarr_metadata.v2.definition import (
    CORE_V2,
    ZarrV2CodecDefinition,
    ZarrV2DataTypeDefinition,
)
from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON
from zarr_metadata.v3._core import CORE_AND_EXTENSIONS
from zarr_metadata.v3._definition import (
    Chunk,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    DataTypeDefinition,
    Lengths,
    StorageTransformerDefinition,
    chunk_grid_lengths,
)
from zarr_metadata.v3._pipeline import Stage, read_pipeline
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
    """One problem per member outside a closed document's declared shape. The kind is `unknown_key`, as a closed TypedDict's checker reports it, so a caller who tolerates a member another writer added can tell it apart from a wrong value."""
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
    """One problem if `doc[key]` is present but not equal to `expected`. The problem is about the type when the value is not of `expected`'s JSON type, and about the value otherwise. The value is compared after refinement, so a `StrEnum` member counts as its string."""
    if key not in doc:
        return ()
    value, found = refine_json(doc[key], (key,))
    if len(found) != 0 or type(value) is not type(expected) or value != expected:
        return (outside_of((key,), doc[key], (expected,)),)
    return ()


def other_members_problems(
    doc: Mapping[object, object],
    standard_keys: frozenset[str],
    *,
    additional_reserved_keys: frozenset[str] = frozenset(),
) -> tuple[ValidationProblem, ...]:
    """Check that every key is a string and every member outside `standard_keys` is a JSON value.

    Use this for a document that allows other members: v3 extension
    fields, and the members that readers of a v2 array ignore.
    """
    return other_members(doc, standard_keys, additional_reserved_keys=additional_reserved_keys)[1]


def other_members(
    doc: Mapping[object, object],
    standard_keys: frozenset[str],
    *,
    additional_reserved_keys: frozenset[str] = frozenset(),
    at: Loc = (),
) -> tuple[dict[str, JSONValue], tuple[ValidationProblem, ...]]:
    """Each member outside `standard_keys`, refined to JSON, and every problem that `other_members_problems` finds. A member that is not JSON is left out of the result."""
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

    `str`, `bytes` and `bytearray` are sequences in Python, but none of
    them is a JSON array. This is a `TypeGuard`, not a `TypeIs`, because
    it returns False for a `str`, which is a `Sequence[object]`.
    """
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_int_sequence(value: object) -> TypeGuard[Sequence[int]]:
    """Whether `value` is a JSON array of integers.

    JSON booleans decode to `bool`, which is an `int` subclass in Python but
    is not an integer in a metadata document, so booleans are excluded.
    This is a `TypeGuard`, not a `TypeIs`, because it returns False for
    `bytes`, which is a `Sequence[int]`.
    """
    return _is_array(value) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def dimension_lengths(
    doc: Mapping[object, object] | Mapping[str, object], key: str
) -> tuple[tuple[int, ...] | None, tuple[ValidationProblem, ...]]:
    """The dimension lengths `doc` holds at `key` (`shape`, `chunks`), and every problem with them.

    Dimension lengths are non-negative integers. The lengths are None when
    `doc` has nothing at `key`, or when the value at `key` has a problem.
    """
    if key not in doc:
        return None, ()
    value, found = refine_json(doc[key], (key,))
    if len(found) != 0:
        # An integer too large to write as JSON text, or a value nested
        # too deep.
        return None, found
    if not _is_int_sequence(value):
        return None, (ValidationProblem((key,), "expected an array of integers", "invalid_type"),)
    if any(item < 0 for item in value):
        return None, (ValidationProblem((key,), "expected non-negative integers", "invalid_value"),)
    return tuple(value), ()


def _is_canonical_dtype_v2(value: object) -> bool:
    """Whether a validated v2 dtype uses the tuple-backed public representation."""
    if isinstance(value, str):
        return True
    if not is_tuple(value):
        return False
    for record in value:
        if not is_tuple(record):
            return False
        fields = record
        if not _is_canonical_dtype_v2(fields[1]):
            return False
        if len(fields) == 3 and not isinstance(fields[2], tuple):
            return False
    return True


def _is_canonical_metadata_field_v3(value: object) -> bool:
    """Whether a validated v3 metadata field has its declared runtime container type."""
    return isinstance(value, (str, dict))


def is_canonical_array_metadata_v3(value: object) -> bool:
    """Whether a validated v3 array document matches `ZarrV3ArrayMetadataJSON` at runtime."""
    if not is_object(value) or not isinstance(value, dict):
        return False
    doc = value
    codecs = doc["codecs"]
    if not isinstance(doc["shape"], tuple) or not is_tuple(codecs):
        return False
    transformers = doc.get("storage_transformers", ())
    if not is_tuple(transformers):
        return False
    if "dimension_names" in doc and not isinstance(doc["dimension_names"], tuple):
        return False
    if not all(_is_canonical_metadata_field_v3(doc[key]) for key, _ in _EXTENSION_POINTS_V3):
        return False
    return all(_is_canonical_metadata_field_v3(item) for item in (*codecs, *transformers))


def _is_canonical_array_metadata_v2(value: object) -> bool:
    """Whether a validated v2 array document matches `ZarrV2ArrayMetadataJSON` at runtime."""
    if not is_object(value) or not isinstance(value, dict):
        return False
    doc = value
    if not isinstance(doc["shape"], tuple) or not isinstance(doc["chunks"], tuple):
        return False
    if not _is_canonical_dtype_v2(doc["dtype"]):
        return False
    compressor = doc["compressor"]
    if compressor is not None and not isinstance(compressor, dict):
        return False
    filters = doc["filters"]
    return filters is None or (
        is_tuple(filters) and all(isinstance(item, dict) for item in filters)
    )


def validate_attributes(value: object) -> tuple[ValidationProblem, ...]:
    """Validate an `attributes` value: a mapping with string keys.

    Returns a problem at `("attributes",)` if it is not, else `()`. Shared by the
    v2 and v3 validators. The loc is relative to the document, not to the
    value, since this function is only called with a document's `attributes`
    value.
    """
    return attributes_of(value)[1]


def members_past_the_levels(
    doc: Mapping[object, object], at: Loc
) -> dict[object, ValidationProblem]:
    """Each member of `doc` (a document located at `at`) that is a container nested deeper than a reader walks, mapped to its problem, located in the document.

    A reader reports these members and does not walk into them. This
    matches what `refine_json` would report for the whole document: a
    document one level before the depth cap can hold scalars, and nothing
    else.
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
    """An `attributes` value refined as user data, and every problem `validate_attributes` finds. The value is None when it is not an object with string keys, or holds a value that is not JSON.

    `at` is the location, within the input document, of the document
    that holds the attributes. Nesting depth is counted from the root of
    the input document. The problems are located relative to the
    document that holds the attributes.
    """
    if not is_object(value) or not all(isinstance(k, str) for k in value):
        return None, (
            ValidationProblem(
                ("attributes",), "expected an object with string keys", "invalid_type"
            ),
        )
    attributes: dict[str, JSONValue] = {}
    problems: list[ValidationProblem] = []
    for key, item in value.items():
        if not isinstance(key, str):
            continue
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
"""The single extension points of a v3 array document, and the kind each is read as: each member that the TypedDict annotates with a field alias."""

_EXTENSION_LISTS_V3: Final[tuple[tuple[str, type[Definition[Any]]], ...]] = tuple(
    (key, kind)
    for key, (annotation, _) in _MEMBERS_V3.items()
    if get_origin(annotation) is tuple and (kind := field_kind(get_args(annotation)[0])) is not None
)
"""The lists of extension points of a v3 array document, and the kind each entry is read as: each member that the TypedDict annotates as a tuple of a field alias."""


@dataclass(frozen=True, slots=True)
class ZarrV3ArrayMetadataReading:
    """A v3 array document as a scope read it, whatever it holds: each extension point, its codecs as a pipeline, every problem, and the model when there are no problems.

    A field the document does not have is `UNSET`. A list of fields that
    the document does not have as a list is empty.
    """

    data_type: ResolvedField[DataTypeDefinition[Any]] | UNSET = UNSET
    """The data type, as the scope read it."""
    chunk_grid: ResolvedField[ChunkGridDefinition[Any]] | UNSET = UNSET
    """The chunk grid, as the scope read it."""
    chunk_key_encoding: ResolvedField[ChunkKeyEncodingDefinition[Any]] | UNSET = UNSET
    """The chunk key encoding, as the scope read it."""
    chunk: Chunk = dataclasses.field(default_factory=Chunk)
    """The chunks the first codec receives: the length of the grid's chunks along each axis of the shape, and the data type."""
    pipeline: tuple[Stage, ...] = ()
    """The codecs, read as a pipeline: each as the scope read it, with the chunk it receives."""
    storage_transformers: tuple[ResolvedField[StorageTransformerDefinition[Any]], ...] = ()
    """The storage transformers, each as the scope read it."""
    problems: tuple[ValidationProblem, ...] = ()
    """Every reason the document is not valid."""
    metadata: ZarrV3ArrayMetadata | None = None
    """The document's model, which holds these fields, when there is no problem; None otherwise."""

    def __reduce__(self) -> str | tuple[object, ...]:
        # A reading that holds a model pickles and copies the same way the
        # model does (as the pair, read once on load). It is restored as
        # that model's own reading, so there is still one model per
        # document.
        if self.metadata is not None:
            return (reading_of, (self.metadata,))
        return object.__reduce__(self)

    def fields(self) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
        """Each field in the document, as the scope read it, with its location in the document.

        The order is: the extension points, then each codec and storage
        transformer at its index. Each is followed by the fields nested in
        it, as `fields_of` returns them: a shard's codecs, or a struct's
        field types. `with_problems` returns each field with its problems.
        """
        own: tuple[tuple[str, ResolvedField[Any] | UNSET], ...] = (
            ("data_type", self.data_type),
            ("chunk_grid", self.chunk_grid),
            ("chunk_key_encoding", self.chunk_key_encoding),
        )
        for key, field in own:
            if field is not UNSET:
                yield from fields_of(field, (key,))
        for index, stage in enumerate(self.pipeline):
            yield from fields_of(stage.codec, ("codecs", index))
        for index, transformer in enumerate(self.storage_transformers):
            yield from fields_of(transformer, ("storage_transformers", index))


class _HoldsReading(Protocol):
    """A model: an object that holds the reading it was built from."""

    @property
    def reading(self) -> object: ...


def reading_of(model: _HoldsReading) -> object:
    """The reading that `model` holds. Unpickling a reading that held a model rebuilds it with this function."""
    return model.reading


@dataclass(frozen=True, slots=True)
class ArrayMembersV3:
    """The members other than fields of a v3 array document that a read found no problems with, refined as the model holds them."""

    shape: tuple[int, ...]
    fill_value: JSONValue
    dimension_names: tuple[str | None, ...] | UNSET
    attributes: dict[str, JSONValue]
    extra_fields: dict[str, JSONValue]
    document: dict[str, JSONValue]
    """The refined document that the model holds: each member as the read refined it, in the order the document writes them."""


@dataclass(frozen=True, slots=True)
class ArrayMembersV2:
    """The members other than fields of a v2 array document that a read found no problems with, refined as the model holds them."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    fill_value: JSONValue
    order: ZarrV2ArrayOrder
    dimension_separator: ZarrV2ArrayDimensionSeparator
    attributes: dict[str, JSONValue] | UNSET
    extra_fields: dict[str, JSONValue]
    document: dict[str, JSONValue]
    """The refined document that the model holds: each member as the read refined it, in the order the document writes them."""


@dataclass(frozen=True, slots=True)
class ZarrV2ArrayMetadataReading:
    """A v2 array document as a scope read it, whatever it holds: its dtype, compressor and filters, every problem, and the model when there are no problems.

    A field the document does not hold is `UNSET`; a `compressor` or
    `filters` written as `null` is None.
    """

    dtype: ResolvedField[ZarrV2DataTypeDefinition[Any]] | UNSET = UNSET
    """The dtype, as the scope read it."""
    compressor: ResolvedField[ZarrV2CodecDefinition[Any]] | UNSET | None = UNSET
    """The compressor, as the scope read it; None when written as `null`."""
    filters: tuple[ResolvedField[ZarrV2CodecDefinition[Any]], ...] | UNSET | None = UNSET
    """The filters, each as the scope read it; None when written as `null`."""
    problems: tuple[ValidationProblem, ...] = ()
    """Every reason the document is not valid."""
    metadata: ZarrV2ArrayMetadata | None = None
    """The document's model, which holds these fields, when there is no problem; None otherwise."""

    def __reduce__(self) -> str | tuple[object, ...]:
        # Same as the v3 reading: pickle through the model, when there is one.
        if self.metadata is not None:
            return (reading_of, (self.metadata,))
        return object.__reduce__(self)

    def fields(self) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
        """Each field in the document, as the scope read it, with its location: the dtype, followed by a struct's record types, then the compressor, then each filter at its index."""
        if self.dtype is not UNSET:
            yield from fields_of(self.dtype, ("dtype",))
        if self.compressor is not UNSET and self.compressor is not None:
            yield from fields_of(self.compressor, ("compressor",))
        if self.filters is not UNSET and self.filters is not None:
            for index, entry in enumerate(self.filters):
                yield from fields_of(entry, ("filters", index))


def read_array_v3(
    value: object, context: Context, *, at: Loc = ()
) -> tuple[ZarrV3ArrayMetadataReading, ArrayMembersV3 | None]:
    """Read `value`, a v3 array document, in `context`, without building its model. Return the reading and the other members, refined; the members are None when the document has a problem.

    `read_array_metadata_v3` builds the model from these two results. A
    field object in the document (an `AcceptedField` built by hand) is
    not JSON, and is refused as such. `at` is the location of this
    document within the input document; a document in consolidated
    metadata is three levels below its group's document. Nesting depth
    is counted from the root of the input document. The problems are
    located relative to this document.
    """
    if not is_object(value):
        return ZarrV3ArrayMetadataReading(problems=not_an_object(value)), None
    doc = value
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
    # Each extension point is read by `resolve`. It first checks the
    # envelope. Every extension *point* must be understood, so a
    # `must_understand` of `false` is refused at each one: ignoring a codec
    # gives wrong bytes just as ignoring a data type gives wrong values.
    # The spec names only the first three
    # (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581),
    # which this reader treats as an oversight, not as permission. Then
    # `resolve` checks the configuration against the definition in
    # `context` that claims the name. `must_understand: false` keeps its
    # meaning where it has one: on an unknown top-level extension *field*,
    # which a reader really can skip.
    # Each member is refined once, here, and the refined value is kept.
    # The document the model holds is built from these values, so nothing
    # is walked twice.
    refined: dict[str, JSONValue] = {
        **extra_fields,
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
    }
    read: dict[str, ResolvedField[Any]] = {}
    for key, kind in _EXTENSION_POINTS_V3:
        if key in doc:
            refined[key], found = refine_json(doc[key], (*at, key))
            read[key], found = resolve_refined(
                doc[key], refined[key], found, kind, context, (*at, key)
            )
            problems.extend(within(found, at))
    # The fill value must be JSON. It is checked against the data type the
    # scope read, if any. If nothing in scope claims the data type, the
    # fill value is not checked.
    fill_value: JSONValue = None
    if "fill_value" in doc:
        fill_value, found = refine_json(doc["fill_value"], (*at, "fill_value"))
        refined["fill_value"] = fill_value
        problems.extend(within(found, at))
        if len(found) == 0 and "data_type" in read:
            problems.extend(fill_value_problems(read["data_type"], fill_value, ("fill_value",)))
    # Once both are read, the chunk grid is checked against the shape. The
    # grid gives the chunk lengths the first codec receives: one entry for
    # each dimension of the shape, None where the length is unknown.
    lengths: Lengths | None = None if shape is None else (None,) * len(shape)
    if "chunk_grid" in read and shape is not None:
        lengths, found = chunk_grid_lengths(read["chunk_grid"], shape, ("chunk_grid",))
        problems.extend(found)
    listed: dict[str, list[ResolvedField[Any]]] = {}
    for key, kind in _EXTENSION_LISTS_V3:
        if key in doc:
            entries = doc[key]
            if not _is_array(entries):
                problems.append(ValidationProblem((key,), "expected an array", "invalid_type"))
            else:
                listed[key] = []
                items: list[JSONValue] = []
                for index, entry in enumerate(entries):
                    item, found = refine_json(entry, (*at, key, index))
                    items.append(item)
                    resolved, found = resolve_refined(
                        entry, item, found, kind, context, (*at, key, index)
                    )
                    listed[key].append(resolved)
                    problems.extend(within(found, at))
                refined[key] = tuple(items)
    # The codecs are read as a pipeline. The first codec receives the
    # grid's chunks, of the array's data type. The codecs are checked in
    # order, each against the chunk it receives. A valid pipeline holds
    # one array -> bytes codec, so it is not empty.
    chunk = Chunk(lengths, read.get("data_type"))
    pipeline: tuple[Stage, ...] = ()
    if "codecs" in listed:
        pipeline, found = read_pipeline(listed["codecs"], chunk, ("codecs",))
        problems.extend(found)
    attributes: dict[str, JSONValue] | None = {}
    if "attributes" in doc:
        attributes, found = attributes_of(doc["attributes"], at)
        refined["attributes"] = attributes
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
    dimension_names: tuple[str | None, ...] | UNSET = UNSET
    if "dimension_names" in doc:
        # An array of strings and nulls, as checked above. Refine it, so
        # each string is as JSON writes it.
        dimension_names = cast("tuple[str | None, ...]", refine_json(doc["dimension_names"])[0])
        refined["dimension_names"] = dimension_names
    members = ArrayMembersV3(
        shape=shape,
        fill_value=fill_value,
        dimension_names=dimension_names,
        attributes=attributes,
        extra_fields=extra_fields,
        document={key: refined[key] for key in cast("Mapping[str, object]", doc)},
    )
    return reading, members


def validate_array_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a valid v3 array document.

    This checks the structure, and reads each extension point through the
    definition that claims its name in `context`. Examples of problems: a
    gzip `level` out of range, or a key that a codec's configuration does
    not declare. The fill value is checked against the data type as
    `context` read it (for example, an `int8` fill value of 300). The
    chunk grid is checked against the shape (for example, a regular grid
    with chunk lengths for two dimensions, over an array with three). The
    codecs are read as a pipeline, in order, each checked against the
    chunk it receives (for example, a `transpose` whose `order` has a
    different number of axes, or a shard that its inner chunks do not
    divide). A shard's inner and index codecs are checked too.
    A name that nothing in `context` claims is not checked, and neither
    is a fill value of that data type. After a codec with such a name,
    the next codec receives a chunk with unknown shape and data type.
    Unknown top-level keys are allowed (they map to `extra_fields`). A
    reader must understand each one that does not declare
    `must_understand: false`; the model reports these as
    `must_understand_fields`. The result equals the `problems` of
    `read_array_metadata_v3`, which also holds what was read to find them.
    """
    scope = scoped(context, CORE_AND_EXTENSIONS)
    return read_array_v3(value, scope)[0].problems


def is_array_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> TypeGuard[ZarrV3ArrayMetadataJSON]:
    """Whether `value` is a v3 array document, written with tuples, that `validate_array_metadata_v3` finds no problems with."""
    scope = scoped(context, CORE_AND_EXTENSIONS)
    return (
        _is_canonical_json(value, finite=False)
        and not validate_array_metadata_v3(value, context=scope)
        and is_canonical_array_metadata_v3(value)
    )


def parse_array_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> ZarrV3ArrayMetadataJSON:
    """Return `value` as `ZarrV3ArrayMetadataJSON`, or raise `MetadataValidationError`."""
    scope = scoped(context, CORE_AND_EXTENSIONS)
    problems = validate_array_metadata_v3(value, context=scope)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV3ArrayMetadataJSON", arrays_to_tuples(value))


def read_array_v2(
    value: object, context: Context, *, at: Loc = ()
) -> tuple[ZarrV2ArrayMetadataReading, ArrayMembersV2 | None]:
    """Read `value`, a v2 array document, once in `context`. Return the reading, and the other members when there are no problems.

    `dtype`, `compressor` and `filters` are read in `context`. A dtype or
    codec that the scope refuses is a problem. One that the scope does
    not claim is not a problem. `fill_value` is checked against the
    dtype the scope read. `compressor` and `filters` are required keys
    that may be `None`. `at` is the location of this document within the
    input document, and prefixes every problem.
    """
    if not is_object(value):
        return ZarrV2ArrayMetadataReading(problems=within(not_an_object(value), at)), None
    doc = value
    # Unlike the group document ("Other keys MUST NOT be present",
    # https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L313), the v2 array document is open: other keys "SHOULD NOT be
    # present within the metadata object and SHOULD be ignored by
    # implementations" (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L91-L92), so a member outside
    # ARRAY_METADATA_STANDARD_KEYS_V2 is not a problem just for being there.
    # Ignored does not mean unchecked: as in v3, the member must be JSON,
    # and its key must be a string.
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
    # Each member is refined once, here, and the refined value is kept.
    # The document the model holds is built from these values, so nothing
    # is walked twice.
    refined: dict[str, JSONValue] = {
        **extra_fields,
        "zarr_format": 2,
        "shape": shape,
        "chunks": chunks,
    }
    dtype: ResolvedField[ZarrV2DataTypeDefinition[Any]] | UNSET = UNSET
    if "dtype" in doc:
        # A typestr is read by its family's definition, and a list of field
        # records as a struct.
        refined["dtype"], found = refine_json(doc["dtype"], ("dtype",))
        dtype, found = resolve_refined(
            doc["dtype"], refined["dtype"], found, ZarrV2DataTypeDefinition, context, ("dtype",)
        )
        problems.extend(found)
    if "order" in doc:
        # Checked after refinement, so a value that only compares equal to
        # "C" is refused, like every other value outside the set.
        order, found = refine_json(doc["order"], ("order",))
        refined["order"] = order
        if len(found) != 0 or order not in ("C", "F"):
            problems.append(outside_of(("order",), doc["order"], ("C", "F")))
    compressor: ResolvedField[ZarrV2CodecDefinition[Any]] | UNSET | None = UNSET
    if "compressor" in doc:
        compressor = None
        refined["compressor"] = None
        if doc["compressor"] is not None:
            refined["compressor"], found = refine_json(doc["compressor"], ("compressor",))
            compressor, found = resolve_refined(
                doc["compressor"],
                refined["compressor"],
                found,
                ZarrV2CodecDefinition,
                context,
                ("compressor",),
            )
            problems.extend(found)
    filters: tuple[ResolvedField[ZarrV2CodecDefinition[Any]], ...] | UNSET | None = UNSET
    if "filters" in doc:
        filters = None
        refined["filters"] = None
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
                read: list[ResolvedField[ZarrV2CodecDefinition[Any]]] = []
                items: list[JSONValue] = []
                for index, item in enumerate(entries):
                    refined_item, found = refine_json(item, ("filters", index))
                    items.append(refined_item)
                    entry, found = resolve_refined(
                        item,
                        refined_item,
                        found,
                        ZarrV2CodecDefinition,
                        context,
                        ("filters", index),
                    )
                    read.append(entry)
                    problems.extend(found)
                filters = tuple(read)
                refined["filters"] = tuple(items)
    if "dimension_separator" in doc:
        separator, found = refine_json(doc["dimension_separator"], ("dimension_separator",))
        refined["dimension_separator"] = separator
        if len(found) != 0 or separator not in (".", "/"):
            problems.append(
                outside_of(("dimension_separator",), doc["dimension_separator"], (".", "/"))
            )
    fill_value: JSONValue = None
    if "fill_value" in doc:
        # The fill value must be JSON. It is checked against the dtype the
        # scope read, if any. If nothing in scope claims the dtype, the
        # fill value is not checked.
        fill_value, found = refine_json(doc["fill_value"], ("fill_value",))
        refined["fill_value"] = fill_value
        problems.extend(found)
        if len(found) == 0 and dtype is not UNSET:
            problems.extend(fill_value_problems(dtype, fill_value, ("fill_value",)))
    attributes: dict[str, JSONValue] | UNSET | None = UNSET
    if "attributes" in doc:
        attributes, found = attributes_of(doc["attributes"])
        refined["attributes"] = attributes
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
        order=cast("ZarrV2ArrayOrder", refined["order"]),
        dimension_separator=cast(
            "ZarrV2ArrayDimensionSeparator", refined.get("dimension_separator", ".")
        ),
        attributes=attributes,
        extra_fields=extra_fields,
        document={key: refined[key] for key in cast("Mapping[str, object]", doc)},
    )
    return reading, members


def validate_array_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a valid v2 array document, read in `context` (`CORE_V2` when none is given).

    `dtype`, `compressor` and `filters` are read in the scope. A dtype or
    codec that the scope refuses is a problem. One that the scope does
    not claim is not a problem. `fill_value` is checked against the
    dtype the scope read.
    """
    return read_array_v2(value, scoped(context, CORE_V2))[0].problems


def is_array_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> TypeGuard[ZarrV2ArrayMetadataJSON]:
    """Whether `value` is a valid v2 array metadata document, read in `context` (`CORE_V2` when none is given)."""
    return (
        _is_canonical_json(value, finite=False)
        and not validate_array_metadata_v2(value, context=context)
        and _is_canonical_array_metadata_v2(value)
    )


def parse_array_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> ZarrV2ArrayMetadataJSON:
    """Return `value` as `ZarrV2ArrayMetadataJSON`, read in `context` (`CORE_V2` when none is given), or raise `MetadataValidationError` with every problem."""
    problems = validate_array_metadata_v2(value, context=context)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast("ZarrV2ArrayMetadataJSON", arrays_to_tuples(value))


def validate_group_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> tuple[ValidationProblem, ...]:
    """Return every reason `value` is not a structurally-valid v2 group doc.

    Validates the in-memory merged form: the `.zgroup` fields plus an
    optional `attributes` mapping folded in from `.zattrs`. A group has no
    fields for a scope to read. `context` is accepted for consistency with
    every other v2 reader.
    """
    scoped(context, CORE_V2)
    if not is_object(value):
        return not_an_object(value)
    doc = value
    problems: list[ValidationProblem] = list(missing_keys(GROUP_METADATA_REQUIRED_KEYS_V2, doc))
    problems.extend(unexpected_keys(GROUP_METADATA_STANDARD_KEYS_V2, doc))
    problems.extend(check_literal(doc, "zarr_format", 2))
    if "attributes" in doc:
        problems.extend(validate_attributes(doc["attributes"]))
    return with_input(problems, doc)


def is_group_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> TypeGuard[ZarrV2GroupMetadataJSON]:
    """Whether `value` is a structurally-valid v2 group metadata document. `context` is accepted for consistency with every other v2 reader."""
    return _is_canonical_json(value, finite=False) and not validate_group_metadata_v2(
        value, context=context
    )


def parse_group_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> ZarrV2GroupMetadataJSON:
    """Return `value` narrowed to `ZarrV2GroupMetadataJSON`, or raise `MetadataValidationError`. `context` is accepted for consistency with every other v2 reader."""
    problems = validate_group_metadata_v2(value, context=context)
    if len(problems) != 0:
        raise MetadataValidationError(problems)
    return cast(ZarrV2GroupMetadataJSON, arrays_to_tuples(value))


StoreKey = TypeVar("StoreKey", bound=str)
"""The key type of a mapping from store keys to bytes: `str`, or the literal keys of one document."""


def load_store_json(mapping: Mapping[StoreKey, bytes], key: str) -> object:
    """Decode the JSON document stored at `key` in `mapping`.

    Returns `object`, not `Any`: what a store holds is unknown until a
    validator says otherwise, and `Any` would let unchecked values flow
    into typed positions silently. Narrow the result with a `parse_*`.

    Decoding uses Python's `json` module, so `NaN`, `Infinity` and
    `-Infinity` are read as the corresponding floats, which matches how
    zarr-python writes attributes. The document's validator decides where
    such values are allowed. Every ingestion failure here
    surfaces as `MetadataValidationError`: a missing store key is a
    `missing_key` problem, a value that is not `bytes` an `invalid_type`
    problem, and undecodable bytes an `invalid_json` problem, rather than
    leaking `KeyError`, `TypeError` or `json.JSONDecodeError` to
    callers.
    """
    # Look up by a `str` key, whatever narrower key type the mapping
    # declares. A key the mapping does not have is simply absent.
    stored = cast("Mapping[str, bytes]", mapping)
    if key not in stored:
        raise MetadataValidationError(
            [ValidationProblem((key,), "missing store key", "missing_key")]
        )
    # Check the annotation at runtime: `json.loads` also decodes a `str`,
    # and raises `TypeError` for most other types.
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
    """Encode a document that its validator has accepted as JSON bytes.

    A non-finite number is written as Python's `json` module writes it
    (`NaN`, `Infinity`, `-Infinity`), which matches how zarr-python
    writes attributes. The validator keeps such numbers out of every
    other place in the document.
    """
    return json.dumps(value, indent=indent, allow_nan=True).encode("utf-8")
