"""The Zarr v3 array: its reader, its reading, and its model."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, cast, get_args, get_origin

from typing_extensions import TypedDict, Unpack

from zarr_metadata._common import JSONValue
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    copied,
    frozen,
    is_array,
    is_object,
    json_text,
    nested_past_the_levels,
    not_an_object,
    refine_json,
    shown_key,
    with_input,
    within,
)
from zarr_metadata._sentinel import UNSET
from zarr_metadata._typed_json import typeddict_keys
from zarr_metadata.definition._context import scoped
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
    member_changes,
    missing_keys,
    reading_of,
    unclaimed_fields_of,
)
from zarr_metadata.v3._common import ZarrV3ExtensionField, ZarrV3MetadataFieldJSON

# A runtime import: `get_type_hints` resolves the public signatures.
from zarr_metadata.v3._context import ZarrV3Context  # noqa: TC001
from zarr_metadata.v3._core import CORE_AND_EXTENSIONS
from zarr_metadata.v3._definition import (
    ArrayExtraFieldDefinition,
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
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.v3._definition import CodecDefinition, ExtraFieldDefinition
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataStoreKey


# The members a v3 array document must have. The members the spec defines
# are `ArrayExtraFieldDefinition.spec_members`; any other is an extra field.
_ARRAY_METADATA_REQUIRED_KEYS_V3: Final[frozenset[str]] = frozenset(
    ZarrV3ArrayMetadataJSON.__required_keys__
)


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


def no_extra_fields() -> Mapping[str, ResolvedField[Any]]:
    """The extra fields of a reading of a document that has none."""
    return MappingProxyType({})


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
    extra_fields: Mapping[str, ResolvedField[ArrayExtraFieldDefinition[Any]]] = dataclasses.field(
        default_factory=no_extra_fields
    )
    """Each member the spec does not define, as the scope read it, by key: an `UnclaimedField` when no definition in scope claims the key."""
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
        # A read-only view does not pickle, so the extra fields go as a dict.
        values = {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}
        return (_array_reading, ({**values, "extra_fields": dict(self.extra_fields)},))

    def fields(self) -> Iterator[tuple[Loc, ResolvedField[Any]]]:
        """Each field in the document, as the scope read it, with its location in the document.

        The order is: the extension points, then each codec and storage
        transformer at its index, then each extra field at its key. Each is
        followed by the fields nested in it, as `fields_of` returns them: a
        shard's codecs, or a struct's field types. `with_problems` returns
        each field with its problems.
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
        for key, extra in self.extra_fields.items():
            yield from fields_of(extra, (key,))


def _array_reading(values: dict[str, Any]) -> ZarrV3ArrayMetadataReading:
    """Rebuild an array reading from the values `ZarrV3ArrayMetadataReading.__reduce__` returns, with its extra fields held read-only."""
    values["extra_fields"] = MappingProxyType(values["extra_fields"])
    return ZarrV3ArrayMetadataReading(**values)


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
    value: object, context: ZarrV3Context, *, at: Loc = ()
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
    problems: list[ValidationProblem] = list(missing_keys(_ARRAY_METADATA_REQUIRED_KEYS_V3, doc))
    past = members_past_the_levels(doc, at)
    problems.extend(past.values())
    whole, doc = doc, {key: item for key, item in doc.items() if key not in past}
    extra_fields, extra, found = read_extra_fields_v3(doc, ArrayExtraFieldDefinition, context, at)
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
        extra_fields=MappingProxyType(extra),
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


def read_extra_fields_v3(
    doc: Mapping[object, object],
    kind: type[ExtraFieldDefinition[Any]],
    context: ZarrV3Context,
    at: Loc,
) -> tuple[dict[str, JSONValue], dict[str, ResolvedField[Any]], tuple[ValidationProblem, ...]]:
    """Read each member of `doc` (a document at `at`) that is not one of the `spec_members` of `kind` as an extra field of `kind`.

    Returns three values: each member that is JSON, refined; each member
    as the scope read it, by key; and every problem, located in `doc`. A
    member that a definition in `context` claims is read by that
    definition, and must be an object whose numbers are finite, unless the
    definition reads user data. Any other member is not read: it is an
    `UnclaimedField`, and, like attributes, need only be JSON as Python's
    `json` module reads it.
    """
    values: dict[str, JSONValue] = {}
    fields: dict[str, ResolvedField[Any]] = {}
    problems: list[ValidationProblem] = []
    for key, value in doc.items():
        if not isinstance(key, str):
            problems.append(
                ValidationProblem((), f"non-string top-level key {shown_key(key)}", "invalid_type")
            )
            continue
        if key in kind.spec_members:
            continue
        member = {key: value}
        definition = context.claimant(kind, key)
        refined, found = kind._refined(member, (*at, key), definition)  # pyright: ignore[reportPrivateUsage]
        if isinstance(refined, dict):
            values[key] = refined[key]
            if definition is None:
                fields[key] = UnclaimedField(json=refined, name=key, read_as=kind)
                continue
        fields[key], found = resolve_refined(member, refined, found, kind, context, (*at, key))
        problems.extend(within(found, at))
    return values, fields, tuple(problems)


def must_understand_subset(
    fields: Mapping[str, AcceptedField[Any] | UnclaimedField],
    values: Mapping[str, ZarrV3ExtensionField],
) -> dict[str, ZarrV3ExtensionField]:
    """The extra fields, by key and as written, that no definition in scope reads and that a reader is obligated to understand.

    Per the v3 spec (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1578), an extension field is implicitly `must_understand:
    True` unless it explicitly says otherwise, and an implementation MUST fail to
    open a group or array carrying fields it does not recognize that are not
    explicitly `must_understand: false`. A field that a definition in scope
    read is recognized. A non-mapping field value cannot carry the explicit
    waiver, so it always requires understanding (the runtime isinstance
    check defends against values looser than the declared
    `ZarrV3ExtensionField`).
    """
    return {
        name: values[name]
        for name, field in fields.items()
        if isinstance(field, UnclaimedField) and not _waived(values[name])
    }


def waived_members(values: Mapping[str, ZarrV3ExtensionField]) -> list[str]:
    """The keys of the extra fields in `values` (by key, as written) that say `"must_understand": false`, which a reader may ignore."""
    return [key for key, value in values.items() if _waived(value)]


def _waived(value: ZarrV3ExtensionField) -> bool:
    """Whether `value`, an extra field as written, is an object that says `"must_understand": false`."""
    return isinstance(value, Mapping) and value.get("must_understand") is False


class ZarrV3ArrayMetadataUpdate(TypedDict, total=False, closed=True):
    """The members `ZarrV3ArrayMetadata.update` replaces: each in the form a document writes it, or `UNSET` to remove a member that a document may omit.

    The members a document may omit are `dimension_names`, `attributes`,
    `storage_transformers`, and any member the spec does not define. Those
    go in `extra_fields`, by key: each one given is set, or removed when
    given as `UNSET`, and the others are kept. A member is given by its own
    name only when the spec defines it, so a misspelled name is refused.
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
    extra_fields: Mapping[str, ZarrV3ExtensionField | UNSET]


class ZarrV3ArrayMetadata(Keyed):
    """A v3 array document, and the scope it was read in.

    The model is this pair. `to_json` returns the document as written,
    refined (arrays as tuples, string keys), and `context` is the scope.
    Every typed member is a view of the reading of the document in the
    scope. `data_type`, `chunk_grid`, `chunk_key_encoding`, and each codec
    and storage transformer are fields as the scope read them: an
    `AcceptedField` (read by the definition that claims its name) or an
    `UnclaimedField`, and so is each member in `extra_fields`, which the
    spec does not define. `shape`, `fill_value`, `dimension_names` and
    `attributes` are the values the read refined. A
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

    zarr_format: Final[Literal[3]] = 3  # noqa: PYI064 - a bare Final does not narrow a union of models
    node_type: Final[Literal["array"]] = "array"  # noqa: PYI064 - as above

    @property
    def claims(self) -> Claims:
        """The claims the reading recorded for each name the document writes, keyed by kind and the name the scope files it under. Computed on first access."""
        try:
            return self._claims
        except AttributeError:
            self._claims = MappingProxyType(claims_of(self._reading.fields()))
            return self._claims

    def __init__(self, document: object, context: ZarrV3Context | None = None) -> None:
        context = scoped(context, CORE_AND_EXTENSIONS)
        reading, members = read_array_v3(document, context)
        if members is None:
            raise MetadataValidationError(reading.problems)
        self._adopt(members.document, context, reading, members)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: ZarrV3Context,
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
        context: ZarrV3Context,
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
            MappingProxyType({key: held(extra) for key, extra in reading.extra_fields.items()}),
        )

    # --- the pair ---------------------------------------------------------

    @property
    def context(self) -> ZarrV3Context:
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
        )

    def __reduce__(self) -> tuple[type[ZarrV3ArrayMetadata], tuple[object, ZarrV3Context]]:
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
    def extra_fields(
        self,
    ) -> Mapping[str, AcceptedField[ArrayExtraFieldDefinition[Any]] | UnclaimedField]:
        """Each member the spec does not define, by key, as the scope read it: an `AcceptedField` when a definition in scope claims the key, and an `UnclaimedField` otherwise."""
        return self._shown[3]

    @property
    def unclaimed_fields(self) -> dict[Loc, UnclaimedField]:
        """Every field nothing in scope claims, at any depth, by its location in the document, except an extra member that says `"must_understand": false` and every field inside it, claimed or not, since a reader may ignore such a member whole.

        This answers whether a reader can open the array: the spec requires
        a reader to refuse a node with a field it does not understand, and
        the scope is what the reader understands. A codec inside a shard is
        located inside the shard's configuration. The model only reports
        these fields; refusing to open is up to the reader.
        """
        return unclaimed_fields_of(self._reading.fields(), waived_members(self._shown[2]))

    @property
    def must_understand_fields(self) -> dict[str, ZarrV3ExtensionField]:
        """The extra fields, by key and as written, that no definition in scope reads and that a reader is obligated to understand.

        Each unclaimed member of `extra_fields` not explicitly waived with
        `must_understand: false` (the spec's implicit-true rule, https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1571-L1578). The scope is what
        the reader recognizes, so a compliant reader MUST fail to open the
        array when this is not empty. The model only reports these fields;
        refusing to open is up to the reader.
        """
        return must_understand_subset(self.extra_fields, self._shown[2])

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
        """Return this model with `members` (as JSON) replacing the document's members, read in this model's own scope. A member set to `UNSET` is removed. A member the spec does not define is given in `extra_fields`.

        Raises `TypeError` for a member the update does not take. Raises
        `MetadataValidationError` when the resulting document has a
        problem. So members that depend on each other must be passed
        together, for example a `shape` with a grid that fits it.
        """
        changes = member_changes(
            members, ZarrV3ArrayMetadataUpdate, ArrayExtraFieldDefinition.spec_members
        )
        document: dict[str, object] = {**self._document, **changes}
        for key, value in changes.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: ZarrV3Context | None = None) -> ZarrV3ArrayMetadata:
        """Return this document read in `context`, whatever that changes: a gain, a loss, or a conflict.

        Raises `MetadataValidationError` when the document has a problem
        in `context`. The existing reading is reused when `context` reads
        every claim identically.
        """
        context = scoped(context, CORE_AND_EXTENSIONS)
        if context.disagreements(self.claims).agrees:
            return self._of(self._document, context, self._reading, self._members)
        return type(self)(self._document, context=context)

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
        context = scoped(context, CORE_AND_EXTENSIONS)
        found = context.disagreements(self.claims)
        if len(found.conflicts) != 0:
            raise ScopeConflictError(located_conflicts(self._reading.fields(), found.conflicts))
        return self.with_context(context)

    def refines(self, other: ZarrV3ArrayMetadata) -> bool:
        """Whether this model holds everything `other` holds. Each field must refine its counterpart, in the order `refines` defines for fields (including the fields nested in each field). Every other member must be the same, with the fill value compared in the spelling of the more informed data type. If that data type rejects the other model's fill value, this is not a refinement."""
        if type(other) is not type(self):
            return False
        if len(self.codecs) != len(other.codecs) or len(self.storage_transformers) != len(
            other.storage_transformers
        ):
            return False
        if self.extra_fields.keys() != other.extra_fields.keys():
            return False
        pairs = (
            (self.data_type, other.data_type),
            (self.chunk_grid, other.chunk_grid),
            (self.chunk_key_encoding, other.chunk_key_encoding),
            *zip(self.codecs, other.codecs, strict=True),
            *zip(self.storage_transformers, other.storage_transformers, strict=True),
            *((self.extra_fields[key], other.extra_fields[key]) for key in self.extra_fields),
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
        **overrides: Unpack[ZarrV3ArrayMetadataUpdate],
    ) -> ZarrV3ArrayMetadata:
        """Return a scalar `uint8` array, with any document members in `overrides` applied, read in `context`. `overrides` takes what `update` takes, including `UNSET` to drop a member.

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
        changes = member_changes(
            overrides, ZarrV3ArrayMetadataUpdate, ArrayExtraFieldDefinition.spec_members
        )
        lengths, _ = dimension_lengths(changes, "shape")
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
        merged = {
            key: value for key, value in {**document, **changes}.items() if value is not UNSET
        }
        return cls(merged, context=context)

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
        extra_fields_key(reading.extra_fields),
    )


def extra_fields_key(extra_fields: Mapping[str, ResolvedField[Any]]) -> tuple[object, ...]:
    """A key for what a document's extra fields mean: each key with its field's `field_key`, sorted by key."""
    return tuple(
        (key, field_key(extra))
        for key, extra in sorted(extra_fields.items(), key=lambda item: item[0])
    )


def read_array_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> ZarrV3ArrayMetadataReading:
    """Read `value`, a v3 array document, in `context`, whatever it holds.

    The result holds everything the read finds: each extension point as
    `context` read it (an `AcceptedField` read by the definition that
    claims its name, an `UnclaimedField`, or a `RefusedField`), the chunks
    the codecs receive, each codec with the chunk it receives, every
    problem, and, when there are no problems, the document's model, which
    holds the same reading.

    The read checks the structure, and reads each extension point through
    the definition that claims its name in `context`. Examples of
    problems: a gzip `level` out of range, or a key that a codec's
    configuration does not declare. The fill value is checked against the
    data type as `context` read it (for example, an `int8` fill value of
    300). The chunk grid is checked against the shape (for example, a
    regular grid with chunk lengths for two dimensions, over an array with
    three). The codecs are read as a pipeline, in order, each checked
    against the chunk it receives (for example, a `transpose` whose
    `order` has a different number of axes, or a shard that its inner
    chunks do not divide). A shard's inner and index codecs are checked
    too. A name that nothing in `context` claims is not checked, and
    neither is a fill value of that data type. After a codec with such a
    name, the next codec receives a chunk with unknown shape and data
    type. Unknown top-level keys are allowed (they map to
    `extra_fields`). A reader must understand each one that does not
    declare `must_understand: false`; the model reports these as
    `must_understand_fields`.

    To apply a policy over the fields (for example, allow only the core
    spec's fields), iterate over the reading's `fields()`. A value that
    is not an object has no fields.
    """
    context = scoped(context, CORE_AND_EXTENSIONS)
    reading, members = read_array_v3(value, context)
    if members is None:
        return reading
    model = ZarrV3ArrayMetadata._of(members.document, context, reading, members)  # pyright: ignore[reportPrivateUsage]
    return model.reading
