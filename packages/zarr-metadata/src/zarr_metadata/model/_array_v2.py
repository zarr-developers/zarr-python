"""The Zarr v2 array: its reader, its reading, and its model."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata._common import JSONValue
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    copied,
    frozen,
    is_array,
    is_object,
    is_tuple,
    json_text,
    not_an_object,
    outside_of,
    refine_json,
    with_input,
    within,
)
from zarr_metadata._sentinel import UNSET
from zarr_metadata._typed_json import typeddict_keys
from zarr_metadata.definition._context import disagreements, scoped
from zarr_metadata.definition._definition import (
    AcceptedField,
    ResolvedField,
    UnclaimedField,
    field_key,
    fields_of,
    fill_value_problems,
    held,
    resolve,
    resolve_refined,
    spelled_canonically,
)
from zarr_metadata.definition._scope import ScopeConflictError, claims_of
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
    other_members,
    reading_of,
    unclaimed_fields_of,
)

# A runtime import: `get_type_hints` resolves the public signatures.
from zarr_metadata.v2._context import ZarrV2Context  # noqa: TC001
from zarr_metadata.v2._field_json import ZarrV2DataTypeMetadata
from zarr_metadata.v2.array import (
    ZARR_V2_ARRAY_METADATA_STORE_KEY,
    ZarrV2ArrayDimensionSeparator,
    ZarrV2ArrayMetadataJSON,
    ZarrV2ArrayOrder,
)
from zarr_metadata.v2.attributes import ZARR_V2_ATTRIBUTES_STORE_KEY
from zarr_metadata.v2.codec import ZarrV2CodecMetadata
from zarr_metadata.v2.definition import (
    CORE_V2,
    ZarrV2CodecDefinition,
    ZarrV2DataTypeDefinition,
)
from zarr_metadata.v2.group import ZARR_V2_GROUP_METADATA_STORE_KEY

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.v2.array import ZarrV2ArrayMetadataStoreKey
    from zarr_metadata.v2.attributes import ZarrV2AttributesStoreKey


_ARRAY_METADATA_REQUIRED_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2ArrayMetadataJSON.__required_keys__
)


_ARRAY_METADATA_OPTIONAL_KEYS_V2: Final[frozenset[str]] = frozenset(
    ZarrV2ArrayMetadataJSON.__optional_keys__
)


_ARRAY_METADATA_STANDARD_KEYS_V2: Final[frozenset[str]] = (
    _ARRAY_METADATA_REQUIRED_KEYS_V2 | _ARRAY_METADATA_OPTIONAL_KEYS_V2
)


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

    A field the document does not hold is `UNSET`, and so is a `filters`
    that is not an array, which holds no filters to read. A `compressor`
    or `filters` written as `null` is None.
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


def read_array_v2(
    value: object, context: ZarrV2Context, *, at: Loc = ()
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
    # _ARRAY_METADATA_STANDARD_KEYS_V2 is not a problem just for being there.
    # Ignored does not mean unchecked: as in v3, the member must be JSON,
    # and its key must be a string.
    problems: list[ValidationProblem] = list(missing_keys(_ARRAY_METADATA_REQUIRED_KEYS_V2, doc))
    extra_fields, found = other_members(doc, _ARRAY_METADATA_STANDARD_KEYS_V2)
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
        entries = doc["filters"]
        if entries is None:
            filters = None
            refined["filters"] = None
        elif not is_array(entries):
            # There are no filters to read, so `filters` stays UNSET: None
            # means that the document writes null.
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


NODE_FILES_V2: Final = (
    ZARR_V2_ARRAY_METADATA_STORE_KEY,
    ZARR_V2_GROUP_METADATA_STORE_KEY,
    ZARR_V2_ATTRIBUTES_STORE_KEY,
)
"""The files that describe a v2 node: `.zarray` or `.zgroup`, and `.zattrs`."""


def decoded_node_files_v2(
    mapping: Mapping[StoreKey, bytes], keys: tuple[str, ...] = NODE_FILES_V2
) -> tuple[dict[str, object], tuple[ValidationProblem, ...]]:
    """The JSON of each of `keys` that `mapping` holds, decoded from its bytes, and the problem of each one that is not bytes of JSON. A key that `mapping` does not hold is left out."""
    stored = cast("Mapping[str, bytes]", mapping)
    files: dict[str, object] = {}
    problems: list[ValidationProblem] = []
    for key in keys:
        if key not in stored:
            continue
        try:
            files[key] = load_store_json(mapping, key)
        except MetadataValidationError as error:
            problems.extend(error.problems)
    return files, tuple(problems)


def node_document_v2(
    files: Mapping[str, object], key: str, beside_attributes: ValidationProblem
) -> tuple[object, tuple[ValidationProblem, ...]]:
    """The merged v2 document of the node whose file is `key` in `files` (each file's decoded JSON, keyed by its name): the file's JSON, with the attributes at `.zattrs` when `files` holds it.

    Returns the document and the problems found in merging. A missing
    `key` gives None and that problem. A file that holds `attributes`
    itself gives `beside_attributes`, since attributes belong in
    `.zattrs`, and the document without that member, so that its reader
    still reports every other problem. A file that is not an object is
    returned as it is, for the document's reader to report.
    """
    if key not in files:
        return None, (ValidationProblem((key,), "missing store key", "missing_key"),)
    document = files[key]
    if not is_object(document):
        return document, ()
    problems: tuple[ValidationProblem, ...] = ()
    if "attributes" in document:
        problems = with_input((beside_attributes,), document)
        document = {name: value for name, value in document.items() if name != "attributes"}
    if ZARR_V2_ATTRIBUTES_STORE_KEY not in files:
        return document, problems
    return {**document, "attributes": files[ZARR_V2_ATTRIBUTES_STORE_KEY]}, problems


ARRAY_FILE_ATTRIBUTES: Final = ValidationProblem(
    ("attributes",),
    f"unexpected key 'attributes' in {ZARR_V2_ARRAY_METADATA_STORE_KEY}: an array's attributes "
    f"belong in {ZARR_V2_ATTRIBUTES_STORE_KEY}",
    "invalid_value",
)
"""The problem of a `.zarray` that holds `attributes`, which belong in `.zattrs`. A `.zarray` may hold members the spec does not define, so the key is not unknown, but this one is taken by the attributes."""


def read_array_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> ZarrV2ArrayMetadataReading:
    """Read `value`, a v2 array document, in `context` (`CORE_V2` when none is given), whatever it holds.

    The result holds everything the read finds: the dtype, the compressor
    and each filter as the scope read them (an `AcceptedField` read by
    the definition that claims the typestr or id, an `UnclaimedField`, or
    a `RefusedField`), every problem, and, when there are no problems,
    the document's model. A dtype or codec that the scope refuses is a
    problem. One that the scope does not claim is not a problem.
    `fill_value` is checked against the dtype the scope read.
    """
    context = scoped(context, CORE_V2)
    reading, members = read_array_v2(value, context)
    if members is None:
        return reading
    model = ZarrV2ArrayMetadata._of(members.document, context, reading, members)  # pyright: ignore[reportPrivateUsage]
    return model.reading


class ZarrV2ArrayMetadataUpdate(TypedDict, total=False, closed=True):
    """The members `ZarrV2ArrayMetadata.update` replaces: each in the form a document writes it, or `UNSET` to remove a member that a document may omit.

    The members a document may omit are `attributes` (meaning no
    `.zattrs`), `dimension_separator` (read as `"."`), and any member the
    spec does not define. Those go in `extra_fields`, by key: each one
    given is set, or removed when given as `UNSET`, and the others are
    kept. A member is given by its own name only when the spec defines it,
    so a misspelled name is refused.
    """

    shape: tuple[int, ...]
    dtype: ZarrV2DataTypeMetadata
    chunks: tuple[int, ...]
    fill_value: JSONValue
    order: ZarrV2ArrayOrder
    compressor: ZarrV2CodecMetadata | None
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    dimension_separator: ZarrV2ArrayDimensionSeparator | UNSET
    attributes: Mapping[str, JSONValue] | UNSET
    extra_fields: Mapping[str, JSONValue | UNSET]


_ARRAY_SPEC_MEMBERS_V2: Final[frozenset[str]] = frozenset(
    {"zarr_format", *typeddict_keys(ZarrV2ArrayMetadataUpdate).members} - {"extra_fields"}
)
"""The members the spec defines in a v2 array document, as the model merges it: these are never given in `extra_fields`."""


class ZarrV2ArrayMetadata(Keyed):
    """A v2 array document, and the scope it was read in.

    Like the v3 models, this model is a pair. `to_json` returns the merged
    document (the `.zarray` members, plus `attributes` when a `.zattrs`
    holds them) as written, refined. A `.zarray` that omits
    `dimension_separator` means `"."` by the v2 convention
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L81-L86):
    the `dimension_separator` property returns `"."`, and the model compares
    equal to one whose document writes `"."`, but `to_json` writes the
    document without the key. `dtype`, `compressor` and
    each filter are views of the reading: an `AcceptedField` read by the
    definition in scope that claims the typestr or id, or an
    `UnclaimedField`. `attributes` is empty when no `.zattrs` exists, as
    in a v3 model. The document still tells the two apart: `to_json` and
    `to_key_value` write `attributes` and `.zattrs` only when a `.zattrs`
    exists, even an empty one. A member the spec does not define is kept
    in `extra_fields`. A model is only built by reading:
    the constructor reads `document` in `context` and raises
    `MetadataValidationError` with every problem, so no model is invalid.
    Two models are equal when their documents mean the same thing in
    their scopes, as the model key defines. `update` reads new members
    in the model's own scope. `with_context` and `refined_in` read the
    document in another scope. A model pickles as its pair.
    """

    __slots__ = ("_context", "_document", "_members", "_reading", "_shown")

    zarr_format: Final[Literal[2]] = 2  # noqa: PYI064 - a bare Final does not narrow a union of models
    node_type: Final[Literal["array"]] = "array"  # noqa: PYI064 - as above

    def __init__(self, document: object, context: ZarrV2Context | None = None) -> None:
        context = scoped(context, CORE_V2)
        reading, members = read_array_v2(document, context)
        if members is None:
            raise MetadataValidationError(reading.problems)
        self._adopt(members.document, context, reading, members)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: ZarrV2Context,
        reading: ZarrV2ArrayMetadataReading,
        members: ArrayMembersV2,
    ) -> ZarrV2ArrayMetadata:
        """Build a model of a document that a read found no problems with, holding that reading, without reading it again. The readers in this package build models through this method; pyright reports these calls as private use."""
        model = object.__new__(cls)
        model._adopt(document, context, reading, members)
        return model

    def _adopt(
        self,
        document: dict[str, JSONValue],
        context: ZarrV2Context,
        reading: ZarrV2ArrayMetadataReading,
        members: ArrayMembersV2,
    ) -> None:
        self._document = document
        self._context = context
        self._reading = dataclasses.replace(reading, metadata=self)
        self._members = members
        # The members the model exposes, read-only at every level.
        self._shown = (
            frozen(members.fill_value),
            frozen({} if members.attributes is UNSET else members.attributes),
            frozen(members.extra_fields),
        )

    # --- the pair ---------------------------------------------------------

    @property
    def context(self) -> ZarrV2Context:
        """The scope the document was read in. `update` reads new members in this scope."""
        return self._context

    @property
    def reading(self) -> ZarrV2ArrayMetadataReading:
        """The document as the scope read it: the dtype, the compressor, and each filter."""
        return self._reading

    def to_json(self) -> ZarrV2ArrayMetadataJSON:
        """The merged document as written, refined. The result shares no objects with the model.

        `attributes` is included when set, even empty. This is not the
        on-disk `.zarray`, which excludes them: `to_key_value` splits the
        document as a store holds it
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330).
        """
        return cast("ZarrV2ArrayMetadataJSON", copied(self._document))

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV2ArrayMetadataStoreKey | ZarrV2AttributesStoreKey, bytes]:
        """The document as a store holds it: `.zarray` without the attributes, and `.zattrs` with the attributes when they are set, even if empty."""
        zarray = {key: value for key, value in self._document.items() if key != "attributes"}
        out: dict[ZarrV2ArrayMetadataStoreKey | ZarrV2AttributesStoreKey, bytes] = {
            ZARR_V2_ARRAY_METADATA_STORE_KEY: dump_store_json(zarray, indent=indent)
        }
        if "attributes" in self._document:
            out[ZARR_V2_ATTRIBUTES_STORE_KEY] = dump_store_json(
                self._document["attributes"], indent=indent
            )
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._document!r}, context={self._context!r})"

    def _key_of(self) -> tuple[object, ...]:
        """The key `==` and `hash` compare for a v2 array model: what its document means.

        The key holds each field's `field_key`, and the fill value as JSON
        text, in its canonical spelling when a definition in scope read the
        dtype. Every other member is included as it is, with JSON members
        as text. No `.zattrs` counts as empty attributes, since both mean
        that the array has none, as in v3.
        """
        members = self._members
        return (
            members.shape,
            members.chunks,
            members.order,
            members.dimension_separator,
            self._fill_value_key(),
            field_key(self.dtype),
            None if self.compressor is None else field_key(self.compressor),
            None if self.filters is None else tuple(field_key(entry) for entry in self.filters),
            json_text({} if members.attributes is UNSET else members.attributes),
            json_text(members.extra_fields),
        )

    def _plain_key(
        self, dtype: AcceptedField[ZarrV2DataTypeDefinition[Any]] | UnclaimedField
    ) -> tuple[object, ...]:
        """The members other than fields that `refines` compares. The fill value is in the spelling of `dtype`, the dtype of the more informed side."""
        members = self._members
        return (
            members.shape,
            members.chunks,
            members.order,
            members.dimension_separator,
            json_text(spelled_canonically(dtype, members.fill_value)),
            json_text({} if members.attributes is UNSET else members.attributes),
            json_text(members.extra_fields),
        )

    def _fill_value_key(self) -> str:
        """The fill value as `==` compares it, as JSON text: in its canonical spelling when a definition in scope read the dtype, and as written otherwise."""
        fill_value = self._members.fill_value
        if isinstance(self.dtype, AcceptedField):
            return json_text(spelled_canonically(self.dtype, fill_value))
        return json_text(fill_value)

    def __reduce__(self) -> tuple[type[ZarrV2ArrayMetadata], tuple[object, ZarrV2Context]]:
        return type(self), (self._document, self._context)

    # --- typed views ------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """The array's shape."""
        return self._members.shape

    @property
    def chunks(self) -> tuple[int, ...]:
        """The shape of each chunk."""
        return self._members.chunks

    @property
    def fill_value(self) -> JSONValue:
        """The fill value as written, refined; read-only at every level."""
        return self._shown[0]

    @property
    def order(self) -> ZarrV2ArrayOrder:
        """The in-chunk layout, `"C"` or `"F"`."""
        return self._members.order

    @property
    def dimension_separator(self) -> ZarrV2ArrayDimensionSeparator:
        """The separator between chunk indices in a chunk key: `"."` when the document writes none."""
        return self._members.dimension_separator

    @property
    def unclaimed_fields(self) -> dict[Loc, UnclaimedField]:
        """Every field nothing in scope claims, its dtype or a codec, located as `reading.fields()` locates it: at its place in the document, except that a struct dtype's field types are located by record and position, as their problems are.

        This answers whether a reader can open the array: a reader cannot
        decode data with a dtype or a codec it does not understand, and the
        scope is what the reader understands. The model only reports these
        fields; refusing to open is up to the reader.
        """
        return unclaimed_fields_of(self._reading.fields(), ())

    @property
    def attributes(self) -> Mapping[str, JSONValue]:
        """The user attributes a `.zattrs` holds, read-only at every level; empty when there is no `.zattrs`."""
        return self._shown[1]

    @property
    def extra_fields(self) -> Mapping[str, JSONValue]:
        """Every member the spec does not define, as written, read-only at every level."""
        return self._shown[2]

    @property
    def dtype(self) -> AcceptedField[ZarrV2DataTypeDefinition[Any]] | UnclaimedField:
        """The dtype as the scope read it: read by its family's definition, or unclaimed."""
        return held(self._reading.dtype)

    @property
    def compressor(self) -> AcceptedField[ZarrV2CodecDefinition[Any]] | UnclaimedField | None:
        """The compressor as the scope read it; None when written as `null`."""
        compressor = self._reading.compressor
        return None if compressor is None else held(compressor)

    @property
    def filters(
        self,
    ) -> tuple[AcceptedField[ZarrV2CodecDefinition[Any]] | UnclaimedField, ...] | None:
        """The filters, each as the scope read it; None when written as `null`."""
        filters = self._reading.filters
        if filters is None:
            return None
        if filters is UNSET:
            msg = "expected filters a read found nothing wrong with, got UNSET"
            raise TypeError(msg)
        return tuple(held(entry) for entry in filters)

    # --- changing ---------------------------------------------------------

    def update(self, **members: Unpack[ZarrV2ArrayMetadataUpdate]) -> ZarrV2ArrayMetadata:
        """Return this model with `members` (as JSON) replacing the document's members, read in this model's own scope. A member set to `UNSET` is removed.

        Raises `MetadataValidationError` when the resulting document has a
        problem. So members that depend on each other must be passed
        together, for example a `dtype` with a fill value of that dtype.
        """
        changes = member_changes(members, ZarrV2ArrayMetadataUpdate, _ARRAY_SPEC_MEMBERS_V2)
        document: dict[str, object] = {**self._document, **changes}
        for key, value in changes.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: ZarrV2Context | None = None) -> ZarrV2ArrayMetadata:
        """Return this document read in `context`, whatever that changes: a gain, a loss, or a conflict.

        Raises `MetadataValidationError` when the document has a problem
        in `context`. The existing reading is reused when `context` reads
        every claim identically.
        """
        context = scoped(context, CORE_V2)
        if disagreements(context, claims_of(self._reading.fields())).agrees:
            return self._of(self._document, context, self._reading, self._members)
        return type(self)(self._document, context=context)

    def refined_in(self, context: ZarrV2Context | None = None) -> ZarrV2ArrayMetadata:
        """Return this document read in `context`. `context` may claim names this scope left unclaimed, but must not contradict any claim.

        Raises `ScopeConflictError` when `context` reads a typestr or id
        with a different definition, or with no definition, where this
        scope read it with one. The error names each one and its location
        in the document. Raises `MetadataValidationError` when a definition
        that `context` claims rejects what was written.
        """
        context = scoped(context, CORE_V2)
        found = disagreements(context, claims_of(self._reading.fields()))
        if len(found.conflicts) != 0:
            raise ScopeConflictError(located_conflicts(self._reading.fields(), found.conflicts))
        return self.with_context(context)

    def refines(self, other: ZarrV2ArrayMetadata) -> bool:
        """Whether this model holds everything `other` holds. Each field must refine its counterpart. A `null` compressor or `null` filters refine only a `null`. Every other member must be the same, with the fill value compared in the spelling of the more informed dtype. If that dtype rejects the other model's fill value, this is not a refinement."""
        if type(other) is not type(self):
            return False
        if (self.compressor is None) != (other.compressor is None):
            return False
        if (self.filters is None) != (other.filters is None):
            return False
        mine = () if self.filters is None else self.filters
        theirs = () if other.filters is None else other.filters
        if len(mine) != len(theirs):
            return False
        pairs = [(self.dtype, other.dtype), *zip(mine, theirs, strict=True)]
        if self.compressor is not None and other.compressor is not None:
            pairs.append((self.compressor, other.compressor))
        if not all(refines_field(one, another) for one, another in pairs):
            return False
        if len(fill_value_problems(self.dtype, other._members.fill_value)) != 0:
            return False
        return self._plain_key(self.dtype) == other._plain_key(self.dtype)

    # --- constructors -----------------------------------------------------

    @classmethod
    def create_default(
        cls, *, context: ZarrV2Context | None = None, **overrides: Unpack[ZarrV2ArrayMetadataUpdate]
    ) -> ZarrV2ArrayMetadata:
        """Return a scalar `|u1` array, with any document members in `overrides` applied, read in `context`.

        Raises `MetadataValidationError` when the resulting document has a
        problem. Overriding `shape` without `chunks` sets `chunks` equal to
        `shape`, so one chunk covers the array. Overriding `chunks` without
        `shape` keeps the scalar default shape, which chunks of another
        rank do not fit. A dtype given without a fill value gets a fill
        value of `0` when its family accepts `0`, and `null` otherwise.
        Every family accepts `null`.
        """
        document: dict[str, object] = {
            "zarr_format": 2,
            "shape": (),
            "chunks": (),
            "dtype": "|u1",
            "fill_value": 0,
            "order": "C",
            "compressor": None,
            "filters": None,
            "dimension_separator": ".",
        }
        given = member_changes(overrides, ZarrV2ArrayMetadataUpdate, _ARRAY_SPEC_MEMBERS_V2)
        if "shape" in given and "chunks" not in given:
            lengths, _ = dimension_lengths(given, "shape")
            if lengths is not None:
                given["chunks"] = lengths
        if "dtype" in given and "fill_value" not in given:
            dtype, _ = resolve(given["dtype"], ZarrV2DataTypeDefinition, scoped(context, CORE_V2))
            if len(fill_value_problems(dtype, 0)) != 0:
                given["fill_value"] = None
        merged = {key: value for key, value in {**document, **given}.items() if value is not UNSET}
        return cls(merged, context=context)

    @classmethod
    def from_json(
        cls, data: object, *, context: ZarrV2Context | None = None
    ) -> ZarrV2ArrayMetadata:
        """Return the model of `data`, a v2 array document with its attributes under `attributes`, read in `context`.

        Raises `MetadataValidationError` with every problem the read finds.
        `read_array_metadata_v2` returns the reading this model is built
        from, and the problems of a document that has any.
        """
        return cls(data, context=context)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: ZarrV2Context | None = None
    ) -> ZarrV2ArrayMetadata:
        """Return the model of the array at `.zarray` in `mapping`, with the attributes at `.zattrs` when that key exists, read in `context`.

        Raises `MetadataValidationError` when `.zarray` is missing, bytes
        are not JSON, `.zarray` holds `attributes`, or the document is not
        valid, with every problem the read finds. Raises `TypeError` for a
        scope of another format before any bytes are decoded.
        """
        context = scoped(context, CORE_V2)
        files, problems = decoded_node_files_v2(
            mapping, (ZARR_V2_ARRAY_METADATA_STORE_KEY, ZARR_V2_ATTRIBUTES_STORE_KEY)
        )
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        document, problems = node_document_v2(
            files, ZARR_V2_ARRAY_METADATA_STORE_KEY, ARRAY_FILE_ATTRIBUTES
        )
        if len(problems) != 0:
            if document is not None:
                problems = (*problems, *read_array_metadata_v2(document, context=context).problems)
            raise MetadataValidationError(problems)
        return cls(document, context=context)
