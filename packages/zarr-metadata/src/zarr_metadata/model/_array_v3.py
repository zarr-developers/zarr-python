"""The Zarr v3 array: its reader, its reading, and its model."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypeGuard, cast, get_args, get_origin

from typing_extensions import TypedDict, Unpack

from zarr_metadata._common import JSONValue
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    copied,
    frozen,
    is_array,
    is_object,
    is_tuple,
    json_text,
    nested_past_the_levels,
    not_an_object,
    refine_json,
    with_input,
    within,
)
from zarr_metadata._json import is_canonical_json as _is_canonical_json
from zarr_metadata._sentinel import UNSET
from zarr_metadata._typed_json import typeddict_keys
from zarr_metadata.definition._context import Context, ZarrV3Context, scoped
from zarr_metadata.definition._definition import (
    AcceptedField,
    Definition,
    ResolvedField,
    UnclaimedField,
    field_key,
    field_kind,
    fields_of,
    fill_value_problems,
    held,
    resolve_refined,
    spelled_canonically,
)
from zarr_metadata.definition._scope import Claims, ScopeConflictError, claims_of
from zarr_metadata.definition._scope import refines as refines_field
from zarr_metadata.model._keyed import Keyed
from zarr_metadata.model._shared import (
    StoreKey,
    attributes_of,
    check_literal,
    dimension_lengths,
    dump_store_json,
    load_store_json,
    located_conflicts,
    missing_keys,
    other_members,
    reading_of,
)
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
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
from zarr_metadata.v3.array import (
    ZARR_V3_ARRAY_METADATA_STORE_KEY,
    ZarrV3ArrayMetadataJSON,
    ZarrV3ExtensionField,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.v3._definition import CodecDefinition
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSONPartial, ZarrV3ArrayMetadataStoreKey


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
            if not is_array(entries):
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
        if not is_array(names):
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


def must_understand_subset(
    extra_fields: Mapping[str, ZarrV3ExtensionField],
) -> dict[str, ZarrV3ExtensionField]:
    """The subset of `extra_fields` the reader is obligated to understand.

    Per the v3 spec (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1578), an extension field is implicitly `must_understand:
    True` unless it explicitly says otherwise, and an implementation MUST fail to
    open a group or array carrying fields it does not recognize that are not
    explicitly `must_understand: false`. A non-mapping field value cannot
    carry the explicit waiver, so it always requires understanding (the
    runtime isinstance check defends against values looser than the declared
    `ZarrV3ExtensionField`).
    """
    return {
        name: value
        for name, value in extra_fields.items()
        if not (isinstance(value, Mapping) and value.get("must_understand") is False)
    }


class ZarrV3ArrayMetadataUpdate(TypedDict, total=False, extra_items=ZarrV3ExtensionField | UNSET):
    """The members `ZarrV3ArrayMetadata.update` replaces: each in the form a document writes it, or `UNSET` to remove a member that a document may omit.

    The members a document may omit are `dimension_names`, `attributes`,
    `storage_transformers`, and any member the spec does not define.
    """

    shape: tuple[int, ...]
    data_type: ZarrV3MetadataFieldJSON
    chunk_grid: ZarrV3MetadataFieldJSON
    chunk_key_encoding: ZarrV3MetadataFieldJSON
    fill_value: JSONValue
    codecs: tuple[ZarrV3MetadataFieldJSON, ...]
    attributes: Mapping[str, JSONValue] | UNSET
    storage_transformers: tuple[ZarrV3MetadataFieldJSON, ...] | UNSET
    dimension_names: tuple[str | None, ...] | UNSET


class ZarrV3ArrayMetadata(Keyed):
    """A v3 array document, and the scope it was read in.

    The model is this pair. `to_json` returns the document as written,
    refined (arrays as tuples, string keys), and `context` is the scope.
    Every typed member is a view of the reading of the document in the
    scope. `data_type`, `chunk_grid`, `chunk_key_encoding`, and each codec
    and storage transformer are fields as the scope read them: an
    `AcceptedField` (read by the definition that claims its name) or an
    `UnclaimedField`. `shape`, `fill_value`, `dimension_names`,
    `attributes` and `extra_fields` are the values the read refined. A
    model is only built by reading: the constructor reads `document` in
    `context` and raises `MetadataValidationError` with every problem, so
    no model is invalid. Two models are equal when their documents mean
    the same thing in their scopes, as the model key defines. The scope
    itself is not compared. `update` reads new members in the model's own
    scope. `with_context` and `refined_in` read the document in another
    scope. A model pickles as its pair, provided the definitions in its
    scope can be pickled, which requires their functions to be defined at
    the top level of a module.
    """

    __slots__ = ("_claims", "_context", "_document", "_members", "_reading", "_shown")

    zarr_format: Final = 3
    node_type: Final = "array"

    @property
    def claims(self) -> Claims:
        """The claims the reading recorded for each name the document writes, keyed by kind and the name the scope files it under. Computed on first access."""
        try:
            return self._claims
        except AttributeError:
            self._claims = MappingProxyType(claims_of(self._reading.fields()))
            return self._claims

    def __init__(self, document: object, context: ZarrV3Context | None = None) -> None:
        scope = scoped(context, CORE_AND_EXTENSIONS)
        reading, members = read_array_v3(document, scope)
        if members is None:
            raise MetadataValidationError(reading.problems)
        self._adopt(members.document, scope, reading, members)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: Context,
        reading: ZarrV3ArrayMetadataReading,
        members: ArrayMembersV3,
    ) -> ZarrV3ArrayMetadata:
        """Build a model of a document that a read found no problems with, holding that reading, without reading it again. The readers in this package build models through this method; pyright reports these calls as private use."""
        model = object.__new__(cls)
        model._adopt(document, context, reading, members)
        return model

    def _adopt(
        self,
        document: dict[str, JSONValue],
        context: Context,
        reading: ZarrV3ArrayMetadataReading,
        members: ArrayMembersV3,
    ) -> None:
        self._document = document
        self._context = context
        # The reading holds this model, in the form `read_array_metadata_v3`
        # returns it, no matter how the model was built.
        self._reading = dataclasses.replace(reading, metadata=self)
        self._members = members
        # The members the model exposes, read-only at every level.
        self._shown = (
            frozen(members.fill_value),
            frozen(members.attributes),
            frozen(members.extra_fields),
        )

    # --- the pair ---------------------------------------------------------

    @property
    def context(self) -> Context:
        """The scope the document was read in. `update` reads new members in this scope."""
        return self._context

    @property
    def reading(self) -> ZarrV3ArrayMetadataReading:
        """The document as the scope read it: each field, the pipeline, and the chunk each codec receives."""
        return self._reading

    def to_json(self) -> ZarrV3ArrayMetadataJSON:
        """The document as written, refined. The result shares no objects with the model."""
        return cast("ZarrV3ArrayMetadataJSON", copied(self._document))

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV3ArrayMetadataStoreKey, bytes]:
        """The document as a store holds it: JSON bytes at `zarr.json`, indented by `indent`.

        `NaN`, `Infinity` and `-Infinity` in `attributes` are written as
        bare tokens, as zarr-python writes them. A strict JSON parser
        rejects these tokens.
        """
        return {ZARR_V3_ARRAY_METADATA_STORE_KEY: dump_store_json(self._document, indent=indent)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._document!r}, context={self._context!r})"

    def _key_of(self) -> tuple[object, ...]:
        """The key `==` and `hash` compare for a v3 array model: what its document means, as computed by `array_key_of`."""
        return array_key_of(self._reading, self._members)

    def _plain_key(
        self, data_type: AcceptedField[DataTypeDefinition[Any]] | UnclaimedField
    ) -> tuple[object, ...]:
        """The members other than fields that `refines` compares. The fill value is in the spelling of `data_type`, the data type of the more informed side."""
        members = self._members
        return (
            members.shape,
            json_text(spelled_canonically(data_type, members.fill_value)),
            members.dimension_names,
            json_text(members.attributes),
            json_text(members.extra_fields),
        )

    def __reduce__(self) -> tuple[type[ZarrV3ArrayMetadata], tuple[object, Context]]:
        # Pickle the pair and read it again on load, so a model's reading
        # always matches its document.
        return type(self), (self._document, self._context)

    # --- typed views ------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """The array's shape."""
        return self._members.shape

    @property
    def fill_value(self) -> JSONValue:
        """The fill value as written, read-only at every level."""
        return self._shown[0]

    @property
    def dimension_names(self) -> tuple[str | None, ...] | UNSET:
        """The dimension names; `UNSET` when the document writes none."""
        return self._members.dimension_names

    @property
    def attributes(self) -> Mapping[str, JSONValue]:
        """The attributes, read-only at every level; empty when the document writes none."""
        return self._shown[1]

    @property
    def extra_fields(self) -> Mapping[str, ZarrV3ExtensionField]:
        """Each member the spec does not define, by name, read-only at every level."""
        return self._shown[2]

    @property
    def must_understand_fields(self) -> dict[str, ZarrV3ExtensionField]:
        """Extra fields the reader is obligated to understand.

        Everything in `extra_fields` not explicitly waived with
        `must_understand: false` (the spec's implicit-true rule, https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1578). A compliant
        reader MUST fail to open the array if this contains any field it does
        not recognize; the model layer only partitions by obligation, since
        recognition is reader-specific.
        """
        return must_understand_subset(self.extra_fields)

    @property
    def data_type(self) -> AcceptedField[DataTypeDefinition[Any]] | UnclaimedField:
        """The data type, as the scope read it."""
        return held(self._reading.data_type)

    @property
    def chunk_grid(self) -> AcceptedField[ChunkGridDefinition[Any]] | UnclaimedField:
        """The chunk grid, as the scope read it."""
        return held(self._reading.chunk_grid)

    @property
    def chunk_key_encoding(self) -> AcceptedField[ChunkKeyEncodingDefinition[Any]] | UnclaimedField:
        """The chunk key encoding, as the scope read it."""
        return held(self._reading.chunk_key_encoding)

    @property
    def codecs(self) -> tuple[AcceptedField[CodecDefinition[Any]] | UnclaimedField, ...]:
        """The codecs, each as the scope read it, in pipeline order."""
        return tuple(held(stage.codec) for stage in self._reading.pipeline)

    @property
    def storage_transformers(
        self,
    ) -> tuple[AcceptedField[StorageTransformerDefinition[Any]] | UnclaimedField, ...]:
        """The storage transformers, each as the scope read it."""
        return tuple(held(entry) for entry in self._reading.storage_transformers)

    # --- changing ---------------------------------------------------------

    def update(self, **members: Unpack[ZarrV3ArrayMetadataUpdate]) -> ZarrV3ArrayMetadata:
        """Return this model with `members` (as JSON) replacing the document's members, read in this model's own scope. A member set to `UNSET` is removed.

        Raises `MetadataValidationError` when the resulting document has a
        problem. So members that depend on each other must be passed
        together, for example a `shape` with a grid that fits it.
        """
        document: dict[str, object] = {**self._document, **members}
        for key, value in members.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: ZarrV3Context | None = None) -> ZarrV3ArrayMetadata:
        """Return this document read in `context`, whatever that changes: a gain, a loss, or a conflict.

        Raises `MetadataValidationError` when the document has a problem
        in `context`. The existing reading is reused when `context` reads
        every claim identically.
        """
        scope = scoped(context, CORE_AND_EXTENSIONS)
        if scope.disagreements(self.claims).agrees:
            return self._of(self._document, scope, self._reading, self._members)
        return type(self)(self._document, context=scope)

    def refined_in(self, context: ZarrV3Context | None = None) -> ZarrV3ArrayMetadata:
        """Return this document read in `context`. `context` may claim names this scope left unclaimed, but must not contradict any claim.

        Raises `ScopeConflictError` when `context` reads a name with a
        different definition, or with no definition, where this scope read
        it with one. A loss of meaning is refused in the same way as a
        conflict. The error names each such name and its location in the
        document. Raises `MetadataValidationError` when a definition that
        `context` newly claims rejects what was written under its name: a
        gain can expose a problem. `with_context` reads the document in
        any scope.
        """
        scope = scoped(context, CORE_AND_EXTENSIONS)
        found = scope.disagreements(self.claims)
        if len(found.conflicts) != 0:
            raise ScopeConflictError(located_conflicts(self._reading.fields(), found.conflicts))
        return self.with_context(scope)

    def refines(self, other: ZarrV3ArrayMetadata) -> bool:
        """Whether this model holds everything `other` holds. Each field must refine its counterpart, in the order `refines` defines for fields (including the fields nested in each field). Every other member must be the same, with the fill value compared in the spelling of the more informed data type. If that data type rejects the other model's fill value, this is not a refinement."""
        if type(other) is not type(self):
            return False
        if len(self.codecs) != len(other.codecs) or len(self.storage_transformers) != len(
            other.storage_transformers
        ):
            return False
        pairs = (
            (self.data_type, other.data_type),
            (self.chunk_grid, other.chunk_grid),
            (self.chunk_key_encoding, other.chunk_key_encoding),
            *zip(self.codecs, other.codecs, strict=True),
            *zip(self.storage_transformers, other.storage_transformers, strict=True),
        )
        if not all(refines_field(mine, theirs) for mine, theirs in pairs):
            return False
        if len(fill_value_problems(self.data_type, other._members.fill_value)) != 0:
            return False
        return self._plain_key(self.data_type) == other._plain_key(self.data_type)

    # --- constructors -----------------------------------------------------

    @classmethod
    def create_default(
        cls,
        *,
        context: ZarrV3Context | None = None,
        **overrides: Unpack[ZarrV3ArrayMetadataJSONPartial],
    ) -> ZarrV3ArrayMetadata:
        """Return a scalar `uint8` array, with any document members in `overrides` applied, read in `context`.

        Raises `MetadataValidationError` when the resulting document has a
        problem. So members that depend on each other must be passed
        together: a data type with a fill value of that type, or a grid
        with the shape it fits. The default codec is `bytes` with little
        `endian`, which accepts a data type of any fixed size. Overriding
        `shape` without `chunk_grid` derives a consistent default grid: one
        regular chunk covering the array (`chunk_shape` equal to `shape`,
        with a length of 1 for a dimension of length 0, since "Chunk sizes
        must be greater than zero",
        https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-grids/regular-grid/index.rst#L40).
        """
        # The grid is derived only from a valid shape. An invalid shape
        # derives no grid, and the read reports it.
        lengths, _ = dimension_lengths(overrides, "shape")
        document: dict[str, object] = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": (),
            "fill_value": 0,
            "data_type": "uint8",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": tuple(max(length, 1) for length in lengths or ())},
            },
            "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
            "chunk_key_encoding": {"name": "default"},
        }
        return cls({**document, **overrides}, context=context)

    @classmethod
    def from_json(
        cls, data: object, *, context: ZarrV3Context | None = None
    ) -> ZarrV3ArrayMetadata:
        """Return the model of `data`, a v3 array document read in `context`.

        Raises `MetadataValidationError` with every problem the read finds.
        `read_array_metadata_v3` returns the reading this model is built
        from, and the problems of a document that has any.
        """
        return cls(data, context=context)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: ZarrV3Context | None = None
    ) -> ZarrV3ArrayMetadata:
        """Return the model of the array document at `zarr.json` in `mapping`, read in `context`.

        Raises `MetadataValidationError` when the key is missing, its bytes
        are not JSON, or the document is not valid.
        """
        return cls(load_store_json(mapping, ZARR_V3_ARRAY_METADATA_STORE_KEY), context=context)


def array_key_of(
    reading: ZarrV3ArrayMetadataReading, members: ArrayMembersV3
) -> tuple[object, ...]:
    """A key for what a v3 array document means, from its `reading` and refined `members`. `==` and `hash` compare this key for the model, and two listings of one node are compared by it.

    The key holds each field's `field_key`, and the fill value as JSON
    text: in its canonical spelling when a definition in scope read the
    data type, and as written otherwise. Every other member is included
    as it is, with JSON members as text.
    """
    data_type = held(reading.data_type)
    fill_value = (
        spelled_canonically(data_type, members.fill_value)
        if isinstance(data_type, AcceptedField)
        else members.fill_value
    )
    return (
        members.shape,
        json_text(fill_value),
        field_key(data_type),
        field_key(held(reading.chunk_grid)),
        tuple(field_key(held(stage.codec)) for stage in reading.pipeline),
        field_key(held(reading.chunk_key_encoding)),
        members.dimension_names,
        json_text(members.attributes),
        tuple(field_key(held(entry)) for entry in reading.storage_transformers),
        json_text(members.extra_fields),
    )


def read_array_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> ZarrV3ArrayMetadataReading:
    """Read `value`, a v3 array document, in `context`, whatever it holds.

    The result holds everything the read finds: each extension point as
    `context` read it (an `AcceptedField` read by the definition that
    claims its name, an `UnclaimedField`, or a `RefusedField`), the chunks
    the codecs receive, each codec with the chunk it receives, every
    problem `validate_array_metadata_v3` finds, and, when there are no
    problems, the document's model, which holds the same reading. To
    apply a policy over the fields (for example, allow only the core
    spec's fields), iterate over the reading's `fields()`. A value that
    is not an object has no fields.
    """
    scope = scoped(context, CORE_AND_EXTENSIONS)
    reading, members = read_array_v3(value, scope)
    if members is None:
        return reading
    model = ZarrV3ArrayMetadata._of(members.document, scope, reading, members)  # pyright: ignore[reportPrivateUsage]
    return model.reading
