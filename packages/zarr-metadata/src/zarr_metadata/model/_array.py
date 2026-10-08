"""In-memory models for Zarr array metadata documents."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata._common import (
    JSONValue,
)
from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    copied,
    frozen,
    is_object,
    json_text,
    refined_object,
    with_input,
)
from zarr_metadata._sentinel import UNSET
from zarr_metadata.model._keyed import Keyed
from zarr_metadata.model._validation import (
    ArrayMembersV2,
    ArrayMembersV3,
    StoreKey,
    ZarrV2ArrayMetadataReading,
    ZarrV3ArrayMetadataReading,
    dimension_lengths,
    dump_store_json,
    load_store_json,
    read_array_v2,
    read_array_v3,
)
from zarr_metadata.v2.array import (
    ZARR_V2_ARRAY_METADATA_STORE_KEY,
    ZarrV2ArrayDimensionSeparator,
    ZarrV2ArrayOrder,
    ZarrV2DataTypeMetadata,
)
from zarr_metadata.v2.attributes import ZARR_V2_ATTRIBUTES_STORE_KEY
from zarr_metadata.v2.codec import ZarrV2CodecMetadata
from zarr_metadata.v2.definition import CORE_V2, resolve_dtype_v2
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3._definition import (
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    Read,
    StorageTransformerDefinition,
    Unclaimed,
    field_key,
    fill_value_problems,
    spelled_canonically,
)
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context
from zarr_metadata.v3._scope import Claims, Conflict, ScopeConflictError, claim_key, claims_of
from zarr_metadata.v3._scope import refines as refines_field
from zarr_metadata.v3.array import ZARR_V3_ARRAY_METADATA_STORE_KEY, ZarrV3ExtensionField

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.v2._definition import ZarrV2CodecDefinition, ZarrV2DataTypeDefinition
    from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON, ZarrV2ArrayMetadataStoreKey
    from zarr_metadata.v2.attributes import ZarrV2AttributesStoreKey
    from zarr_metadata.v3._definition import Resolved
    from zarr_metadata.v3.array import (
        ZarrV3ArrayMetadataJSON,
        ZarrV3ArrayMetadataJSONPartial,
        ZarrV3ArrayMetadataStoreKey,
    )


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
    """The members `ZarrV3ArrayMetadata.update` puts in place: each as a document writes it, or `UNSET` to leave out one a document may leave out.

    Those are `dimension_names`, `attributes`, `storage_transformers`, and
    a member the spec does not define.
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

    The model is the pair: `to_json` is the document as written, refined
    -- arrays as tuples, string keys -- and `context` the scope. Every
    typed member is a view of the reading the pair gives: `data_type`,
    `chunk_grid`, `chunk_key_encoding`, each codec and storage transformer
    as the scope read it, `Read` by the definition that claims its name or
    `Unclaimed`; `shape`, `fill_value`, `dimension_names`, `attributes` and
    `extra_fields` as the read refined them. Built only by reading: the
    constructor reads `document` in `context` and raises
    `MetadataValidationError` with every problem, so no model is invalid.
    Two models are equal when their documents mean the same in their
    scopes, as its key says; the scope itself takes no part. `update`
    reads new members in the model's own scope; `with_context` and
    `refined_in` read the document in another. A model pickles as its
    pair, when the definitions its scope holds do: ones whose functions
    are defined at a module's top level.
    """

    __slots__ = ("_claims", "_context", "_document", "_members", "_reading", "_shown")

    zarr_format: Final = 3
    node_type: Final = "array"

    @property
    def claims(self) -> Claims:
        """What the reading claimed of each name the document writes, keyed as the scope files it."""
        return self._claims

    def __init__(self, document: object, context: Context | None = None) -> None:
        scope = CORE_AND_EXTENSIONS if context is None else context
        reading, members = read_array_v3(document, scope)
        if members is None:
            raise MetadataValidationError(reading.problems)
        self._adopt(refined_object(document), scope, reading, members)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: Context,
        reading: ZarrV3ArrayMetadataReading,
        members: ArrayMembersV3,
    ) -> ZarrV3ArrayMetadata:
        """A model of a document a read found nothing wrong with, holding that reading: no second read. The readers of this package build models through this, the private use pyright reports."""
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
        # The reading holds the model it built, as `read_array_metadata_v3`
        # hands it back, however the model was built.
        self._reading = dataclasses.replace(reading, metadata=self)
        self._members = members
        # What the model shows of its members, read-only at every level.
        self._shown = (
            frozen(members.fill_value),
            frozen(members.attributes),
            frozen(members.extra_fields),
        )
        self._key = self._key_of()
        self._claims = MappingProxyType(claims_of(reading.fields()))

    # --- the pair ---------------------------------------------------------

    @property
    def context(self) -> Context:
        """The scope the document was read in, which `update` reads new members in."""
        return self._context

    @property
    def reading(self) -> ZarrV3ArrayMetadataReading:
        """The document as the scope read it: each field, the pipeline, the chunk each codec is handed."""
        return self._reading

    def to_json(self) -> ZarrV3ArrayMetadataJSON:
        """The document as written, refined, sharing nothing with the model."""
        return cast("ZarrV3ArrayMetadataJSON", copied(self._document))

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV3ArrayMetadataStoreKey, bytes]:
        """The document as a store holds it: JSON bytes at `zarr.json`, indented by `indent`.

        `NaN`, `Infinity` and `-Infinity` in `attributes` are written as
        those bare tokens, as zarr-python writes them, which a strict JSON
        parser refuses.
        """
        return {ZARR_V3_ARRAY_METADATA_STORE_KEY: dump_store_json(self._document, indent=indent)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._document!r}, context={self._context!r})"

    def _key_of(self) -> tuple[object, ...]:
        """What `==` and `hash` compare of a v3 array model: what its document means.

        Each field by its `field_key`, the fill value in its canonical spelling
        as JSON text when a definition in scope read the data type, and every
        other member as it is, the JSON ones as text.
        """
        members = self._members
        return (
            members.shape,
            self._fill_value_key(),
            field_key(self.data_type),
            field_key(self.chunk_grid),
            tuple(field_key(codec) for codec in self.codecs),
            field_key(self.chunk_key_encoding),
            members.dimension_names,
            json_text(members.attributes),
            tuple(field_key(transformer) for transformer in self.storage_transformers),
            json_text(members.extra_fields),
        )

    def _plain_key(
        self, data_type: Read[DataTypeDefinition[Any]] | Unclaimed
    ) -> tuple[object, ...]:
        """What `refines` compares of a model other than its fields, the fill value spelled as `data_type` -- the more informed side's -- spells it."""
        members = self._members
        return (
            members.shape,
            json_text(spelled_canonically(data_type, members.fill_value)),
            members.dimension_names,
            json_text(members.attributes),
            json_text(members.extra_fields),
        )

    def _fill_value_key(self) -> str:
        """What `==` compares of the fill value: its canonical spelling as JSON text when a definition in scope read the data type, and the fill value as written when none did."""
        fill_value = self._members.fill_value
        if isinstance(self.data_type, Read):
            return json_text(spelled_canonically(self.data_type, fill_value))
        return json_text(fill_value)

    def __reduce__(self) -> tuple[type[ZarrV3ArrayMetadata], tuple[object, Context]]:
        # The pair, read again on load: a model's reading never disagrees
        # with its document.
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
    def data_type(self) -> Read[DataTypeDefinition[Any]] | Unclaimed:
        """The data type, as the scope read it."""
        return cast("Read[DataTypeDefinition[Any]] | Unclaimed", self._reading.data_type)

    @property
    def chunk_grid(self) -> Read[ChunkGridDefinition[Any]] | Unclaimed:
        """The chunk grid, as the scope read it."""
        return cast("Read[ChunkGridDefinition[Any]] | Unclaimed", self._reading.chunk_grid)

    @property
    def chunk_key_encoding(self) -> Read[ChunkKeyEncodingDefinition[Any]] | Unclaimed:
        """The chunk key encoding, as the scope read it."""
        return cast(
            "Read[ChunkKeyEncodingDefinition[Any]] | Unclaimed", self._reading.chunk_key_encoding
        )

    @property
    def codecs(self) -> tuple[Read[CodecDefinition[Any]] | Unclaimed, ...]:
        """The codecs, each as the scope read it, in pipeline order."""
        return tuple(
            cast("Read[CodecDefinition[Any]] | Unclaimed", stage.codec)
            for stage in self._reading.pipeline
        )

    @property
    def storage_transformers(
        self,
    ) -> tuple[Read[StorageTransformerDefinition[Any]] | Unclaimed, ...]:
        """The storage transformers, each as the scope read it."""
        return cast(
            "tuple[Read[StorageTransformerDefinition[Any]] | Unclaimed, ...]",
            self._reading.storage_transformers,
        )

    # --- changing ---------------------------------------------------------

    def update(self, **members: Unpack[ZarrV3ArrayMetadataUpdate]) -> ZarrV3ArrayMetadata:
        """This model with `members`, JSON, in place of the document's, `UNSET` leaving one out, read in this model's own scope.

        `MetadataValidationError` when the document they make has a
        problem, so members that go together are passed together: a
        `shape` with a grid that fits it.
        """
        document: dict[str, object] = {**self._document, **members}
        for key, value in members.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: Context | None = None) -> ZarrV3ArrayMetadata:
        """This document read in `context`, whatever that changes: a gain, a loss, a conflict.

        `MetadataValidationError` when the document has a problem there.
        The reading is kept when `context` reads every claim identically.
        """
        scope = CORE_AND_EXTENSIONS if context is None else context
        if scope.disagreements(self._claims).agrees:
            return self._of(self._document, scope, self._reading, self._members)
        return type(self)(self._document, context=scope)

    def refined_in(self, context: Context | None = None) -> ZarrV3ArrayMetadata:
        """This document read in `context`, which may claim what this scope left unclaimed and contradict nothing.

        `ScopeConflictError` naming each name `context` reads by another
        definition, or by none, where this scope read it by one -- a loss
        of meaning is refused as a conflict is -- and where each sits in
        the document. `MetadataValidationError` when a name `context`
        claims refuses what was written under it: a gain can surface a
        problem. `with_context` reads the document in any scope.
        """
        scope = CORE_AND_EXTENSIONS if context is None else context
        found = scope.disagreements(self._claims)
        if len(found.conflicts) != 0:
            raise ScopeConflictError(located_conflicts(self._reading.fields(), found.conflicts))
        return self.with_context(scope)

    def refines(self, other: ZarrV3ArrayMetadata) -> bool:
        """Whether this model holds everything `other` holds: each field refines its counterpart, as `refines` orders fields -- the fields a field holds with it -- and every other member is the same, the fill value as the more informed data type spells it; a fill value that data type refuses is no refinement."""
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
        context: Context | None = None,
        **overrides: Unpack[ZarrV3ArrayMetadataJSONPartial],
    ) -> ZarrV3ArrayMetadata:
        """A scalar `uint8` array, or the one `overrides`, members of its document, make of it, read in `context`.

        `MetadataValidationError` when the document they make has a
        problem, so members that go together are passed together: a data
        type with a fill value of it, a grid with the shape it fits. The
        default codec is `bytes` with a little `endian`, which takes a data
        type of any fixed size. Overriding `shape` without `chunk_grid`
        derives a consistent default grid: one regular chunk covering the
        array (`chunk_shape` equal to `shape`, with a length of 1 for a
        dimension of length 0, since "Chunk sizes must be greater than
        zero",
        https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-grids/regular-grid/index.rst#L40).
        """
        # The grid derives from a shape the read takes; one it refuses is
        # reported by the read, and derives nothing.
        lengths, _ = dimension_lengths(cast("Mapping[object, object]", overrides), "shape")
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
    def from_json(cls, data: object, *, context: Context | None = None) -> ZarrV3ArrayMetadata:
        """The model of `data`, a v3 array document read in `context`.

        `MetadataValidationError` with every problem the read finds.
        `read_array_metadata_v3` gives the reading this model is built
        from, and the problems of a document with some.
        """
        return cls(data, context=context)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: Context | None = None
    ) -> ZarrV3ArrayMetadata:
        """The model of the array document at `zarr.json` in `mapping`, read in `context`.

        `MetadataValidationError` when the key is missing, its bytes are not
        JSON, or the document is not valid.
        """
        return cls(load_store_json(mapping, ZARR_V3_ARRAY_METADATA_STORE_KEY), context=context)


def located_conflicts(
    fields: Iterable[tuple[Loc, Resolved[Any]]], conflicts: Sequence[Conflict]
) -> tuple[Conflict, ...]:
    """Each of `conflicts`, found against a reading's claims, once for each place among `fields` the name it is about sits: located, as a problem is."""
    located: list[Conflict] = []
    placed = list(fields)
    for conflict in conflicts:
        places = [loc for loc, field in placed if claim_key(field) == conflict.key]
        if len(places) == 0:
            located.append(conflict)
        located.extend(dataclasses.replace(conflict, loc=loc) for loc in places)
    return tuple(located)


def read_array_metadata_v3(
    value: object, *, context: Context | None = None
) -> ZarrV3ArrayMetadataReading:
    """`value`, a v3 array document, as `context` read it, whatever it holds.

    Everything a read finds, in one: each extension point as `context`
    read it -- `Read` by the definition that claims its name, `Unclaimed`,
    or `Refused` -- the chunks the codecs are handed, each codec with the
    chunk it is handed, every problem `validate_array_metadata_v3` finds,
    and, when there is none, the document's model, holding the same
    reading. A policy over the fields, the core spec's alone, say, is a
    walk over its `fields()`. A value that is not an object holds no
    field.
    """
    scope = CORE_AND_EXTENSIONS if context is None else context
    reading, members = read_array_v3(value, scope)
    if members is None:
        return reading
    document = refined_object(value)
    model = ZarrV3ArrayMetadata._of(document, scope, reading, members)  # pyright: ignore[reportPrivateUsage]
    return model.reading


def read_array_metadata_v2(
    value: object, *, context: Context | None = None
) -> ZarrV2ArrayMetadataReading:
    """`value`, a v2 array document, as `context` read it, `CORE_V2` when none is given, whatever it holds.

    Everything a read finds, in one: the dtype, the compressor and each
    filter as the scope read them -- `Read` by the definition that claims
    the typestr or id, `Unclaimed`, or `Refused` -- every problem
    `validate_array_metadata_v2` finds, and, when there is none, the
    document's model.
    """
    scope = CORE_V2 if context is None else context
    reading, members = read_array_v2(value, scope)
    if members is None:
        return reading
    document = refined_object(value)
    if "dimension_separator" not in document:
        document = {**document, "dimension_separator": "."}
    model = ZarrV2ArrayMetadata._of(document, scope, reading, members)  # pyright: ignore[reportPrivateUsage]
    return model.reading


class ZarrV2ArrayMetadataUpdate(TypedDict, total=False, extra_items=JSONValue | UNSET):
    """The members `ZarrV2ArrayMetadata.update` puts in place: each as a document writes it, or `UNSET` to leave out one a document may leave out.

    Those are `attributes` (no `.zattrs`), `dimension_separator` (read as
    `"."`), and a member the spec does not define.
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


class ZarrV2ArrayMetadata(Keyed):
    """A v2 array document, and the scope it was read in.

    The pair, as the v3 models are: `to_json` is the merged document --
    the `.zarray` members, and `attributes` when a `.zattrs` holds them --
    as written, refined, with one spelling put in: a `.zarray` that omits
    `dimension_separator` means `"."` by the v2 convention
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L81-L86),
    which the model holds and writes. `dtype`, `compressor` and each
    filter are views of the reading: `Read` by the definition in scope
    that claims the typestr or id, or `Unclaimed`. `attributes` is
    `UNSET` when no `.zattrs` exists, distinct from an empty one. A
    member the spec does not define is kept, in `extra_fields`. Built
    only by reading: the constructor reads `document` in `context` and
    raises `MetadataValidationError` with every problem, so no model is
    invalid. Two models are equal when their documents mean the same in
    their scopes, as its key says. `update` reads new members in
    the model's own scope; `with_context` and `refined_in` read the
    document in another. A model pickles as its pair.
    """

    __slots__ = ("_claims", "_context", "_document", "_members", "_reading", "_shown")

    zarr_format: Final = 2

    @property
    def claims(self) -> Claims:
        """What the reading claimed of each typestr and codec id the document writes, keyed as the scope files them."""
        return self._claims

    def __init__(self, document: object, context: Context | None = None) -> None:
        scope = CORE_V2 if context is None else context
        reading, members = read_array_v2(document, scope)
        if members is None:
            raise MetadataValidationError(reading.problems)
        held = refined_object(document)
        if "dimension_separator" not in held:
            held = {**held, "dimension_separator": "."}
        self._adopt(held, scope, reading, members)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: Context,
        reading: ZarrV2ArrayMetadataReading,
        members: ArrayMembersV2,
    ) -> ZarrV2ArrayMetadata:
        """A model of a document a read found nothing wrong with, holding that reading: no second read. The readers of this package build models through this, the private use pyright reports."""
        model = object.__new__(cls)
        model._adopt(document, context, reading, members)
        return model

    def _adopt(
        self,
        document: dict[str, JSONValue],
        context: Context,
        reading: ZarrV2ArrayMetadataReading,
        members: ArrayMembersV2,
    ) -> None:
        self._document = document
        self._context = context
        self._reading = dataclasses.replace(reading, metadata=self)
        self._members = members
        # What the model shows of its members, read-only at every level.
        self._shown = (
            frozen(members.fill_value),
            UNSET if members.attributes is UNSET else frozen(members.attributes),
            frozen(members.extra_fields),
        )
        self._key = self._key_of()
        self._claims = MappingProxyType(claims_of(reading.fields()))

    # --- the pair ---------------------------------------------------------

    @property
    def context(self) -> Context:
        """The scope the document was read in, which `update` reads new members in."""
        return self._context

    @property
    def reading(self) -> ZarrV2ArrayMetadataReading:
        """The document as the scope read it: the dtype, the compressor, each filter."""
        return self._reading

    def to_json(self) -> ZarrV2ArrayMetadataJSON:
        """The merged document as written, refined, sharing nothing with the model.

        `attributes` is included when set, even empty. This is not the
        on-disk `.zarray`, which excludes them: `to_key_value` splits the
        document as a store holds it
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330).
        """
        return cast("ZarrV2ArrayMetadataJSON", copied(self._document))

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV2ArrayMetadataStoreKey | ZarrV2AttributesStoreKey, bytes]:
        """The document as a store holds it: `.zarray` without the attributes, and `.zattrs` with them when they are set, even empty."""
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
        """What `==` and `hash` compare of a v2 array model: what its document means.

        Each field by its `field_key`, the fill value in its canonical spelling
        as JSON text when a definition in scope read the dtype, and every other
        member as it is, the JSON ones as text; `attributes` as `UNSET` when
        there is no `.zattrs`.
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
            UNSET if members.attributes is UNSET else json_text(members.attributes),
            json_text(members.extra_fields),
        )

    def _plain_key(
        self, dtype: Read[ZarrV2DataTypeDefinition[Any]] | Unclaimed
    ) -> tuple[object, ...]:
        """What `refines` compares of a model other than its fields, the fill value spelled as `dtype` -- the more informed side's -- spells it."""
        members = self._members
        return (
            members.shape,
            members.chunks,
            members.order,
            members.dimension_separator,
            json_text(spelled_canonically(dtype, members.fill_value)),
            UNSET if members.attributes is UNSET else json_text(members.attributes),
            json_text(members.extra_fields),
        )

    def _fill_value_key(self) -> str:
        """What `==` compares of the fill value: its canonical spelling as JSON text when a definition in scope read the dtype, and the fill value as written when none did."""
        fill_value = self._members.fill_value
        if isinstance(self.dtype, Read):
            return json_text(spelled_canonically(self.dtype, fill_value))
        return json_text(fill_value)

    def __reduce__(self) -> tuple[type[ZarrV2ArrayMetadata], tuple[object, Context]]:
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
        """What joins the chunk indices in a key: `"."` when the document writes none."""
        return self._members.dimension_separator

    @property
    def attributes(self) -> Mapping[str, JSONValue] | UNSET:
        """The user attributes a `.zattrs` holds, read-only at every level; `UNSET` when there is no `.zattrs`."""
        return self._shown[1]

    @property
    def extra_fields(self) -> Mapping[str, JSONValue]:
        """Every member the spec does not define, as written, read-only at every level."""
        return self._shown[2]

    @property
    def dtype(self) -> Read[ZarrV2DataTypeDefinition[Any]] | Unclaimed:
        """The dtype as the scope read it: by its family's definition, or unclaimed."""
        return cast("Read[ZarrV2DataTypeDefinition[Any]] | Unclaimed", self._reading.dtype)

    @property
    def compressor(self) -> Read[ZarrV2CodecDefinition[Any]] | Unclaimed | None:
        """The compressor as the scope read it; None when written as `null`."""
        return cast("Read[ZarrV2CodecDefinition[Any]] | Unclaimed | None", self._reading.compressor)

    @property
    def filters(self) -> tuple[Read[ZarrV2CodecDefinition[Any]] | Unclaimed, ...] | None:
        """The filters, each as the scope read it; None when written as `null`."""
        return cast(
            "tuple[Read[ZarrV2CodecDefinition[Any]] | Unclaimed, ...] | None",
            self._reading.filters,
        )

    # --- changing ---------------------------------------------------------

    def update(self, **members: Unpack[ZarrV2ArrayMetadataUpdate]) -> ZarrV2ArrayMetadata:
        """This model with `members`, JSON, in place of the document's, `UNSET` leaving one out, read in this model's own scope.

        `MetadataValidationError` when the document they make has a
        problem, so members that go together are passed together: a
        `dtype` with a fill value of it.
        """
        document: dict[str, object] = {**self._document, **members}
        for key, value in members.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: Context | None = None) -> ZarrV2ArrayMetadata:
        """This document read in `context`, whatever that changes: a gain, a loss, a conflict.

        `MetadataValidationError` when the document has a problem there.
        The reading is kept when `context` reads every claim identically.
        """
        scope = CORE_V2 if context is None else context
        if scope.disagreements(self._claims).agrees:
            return self._of(self._document, scope, self._reading, self._members)
        return type(self)(self._document, context=scope)

    def refined_in(self, context: Context | None = None) -> ZarrV2ArrayMetadata:
        """This document read in `context`, which may claim what this scope left unclaimed and contradict nothing.

        `ScopeConflictError` naming each typestr or id `context` reads by
        another definition, or by none, where this scope read it by one,
        and where each sits in the document. `MetadataValidationError`
        when a definition `context` claims refuses what was written.
        """
        scope = CORE_V2 if context is None else context
        found = scope.disagreements(self._claims)
        if len(found.conflicts) != 0:
            raise ScopeConflictError(located_conflicts(self._reading.fields(), found.conflicts))
        return self.with_context(scope)

    def refines(self, other: ZarrV2ArrayMetadata) -> bool:
        """Whether this model holds everything `other` holds: each field refines its counterpart, a `null` compressor or filters only a `null`, and every other member is the same, the fill value as the more informed dtype spells it; a fill value that dtype refuses is no refinement."""
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
        cls, *, context: Context | None = None, **overrides: Unpack[ZarrV2ArrayMetadataUpdate]
    ) -> ZarrV2ArrayMetadata:
        """A scalar `|u1` array, or the one `overrides`, members of its document, make of it, read in `context`.

        `MetadataValidationError` when the document they make has a
        problem. Overriding `shape` without `chunks` derives `chunks`
        equal to `shape`, one chunk covering the array; overriding `chunks`
        without `shape` keeps the scalar default shape, which chunks of
        another rank do not fit. A dtype given without a fill value takes
        `0` when its family takes it, and `null` otherwise, which every
        family takes.
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
        given: dict[str, object] = dict(overrides)
        if "shape" in given and "chunks" not in given:
            lengths, _ = dimension_lengths(cast("Mapping[object, object]", given), "shape")
            if lengths is not None:
                given["chunks"] = lengths
        if "dtype" in given and "fill_value" not in given:
            dtype, _ = resolve_dtype_v2(given["dtype"], context)
            if len(fill_value_problems(dtype, 0)) != 0:
                given["fill_value"] = None
        merged = {key: value for key, value in {**document, **given}.items() if value is not UNSET}
        return cls(merged, context=context)

    @classmethod
    def from_json(cls, data: object, *, context: Context | None = None) -> ZarrV2ArrayMetadata:
        """The model of `data`, a v2 array document with its attributes under `attributes`, read in `context`.

        `MetadataValidationError` with every problem the read finds.
        `read_array_metadata_v2` gives the reading this model is built
        from, and the problems of a document with some.
        """
        return cls(data, context=context)

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: Context | None = None
    ) -> ZarrV2ArrayMetadata:
        """The model of the array at `.zarray` in `mapping`, with the attributes at `.zattrs` when there is one, read in `context`.

        `MetadataValidationError` when `.zarray` is missing, bytes are not
        JSON, `.zarray` holds `attributes`, or the document is not valid.
        """
        zarray_raw = load_store_json(mapping, ZARR_V2_ARRAY_METADATA_STORE_KEY)
        if not is_object(zarray_raw):
            return cls(zarray_raw, context=context)
        zarray = zarray_raw
        if "attributes" in zarray:
            refused = ValidationProblem(
                ("attributes",), "unexpected document member", "invalid_value"
            )
            raise MetadataValidationError(with_input((refused,), zarray))
        if ZARR_V2_ATTRIBUTES_STORE_KEY in mapping:
            zattrs = load_store_json(mapping, ZARR_V2_ATTRIBUTES_STORE_KEY)
            return cls({**zarray, "attributes": zattrs}, context=context)
        return cls(zarray, context=context)
