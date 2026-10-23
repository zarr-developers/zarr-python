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
    AcceptedField,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    StorageTransformerDefinition,
    UnclaimedField,
    field_key,
    fill_value_problems,
    held,
    spelled_canonically,
)
from zarr_metadata.v3._registry import (
    CORE_AND_EXTENSIONS,
    Context,
    ZarrV2Context,
    ZarrV3Context,
    scoped,
)
from zarr_metadata.v3._scope import Claims, Conflict, ScopeConflictError, claim_key, claims_of
from zarr_metadata.v3._scope import refines as refines_field
from zarr_metadata.v3.array import ZARR_V3_ARRAY_METADATA_STORE_KEY, ZarrV3ExtensionField

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.v2._definition import ZarrV2CodecDefinition, ZarrV2DataTypeDefinition
    from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON, ZarrV2ArrayMetadataStoreKey
    from zarr_metadata.v2.attributes import ZarrV2AttributesStoreKey
    from zarr_metadata.v3._definition import ResolvedField
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


def located_conflicts(
    fields: Iterable[tuple[Loc, ResolvedField[Any]]], conflicts: Sequence[Conflict]
) -> tuple[Conflict, ...]:
    """Locate each of `conflicts` (found against a reading's claims) in the document, as a problem is located. A conflict appears once for each place in `fields` where its name occurs."""
    located: list[Conflict] = []
    placed = list(fields)
    for conflict in conflicts:
        places = [(loc, field) for loc, field in placed if claim_key(field) == conflict.key]
        if len(places) == 0:
            located.append(conflict)
        located.extend(
            dataclasses.replace(conflict, loc=loc, written=field.name) for loc, field in places
        )
    return tuple(located)


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


def read_array_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> ZarrV2ArrayMetadataReading:
    """Read `value`, a v2 array document, in `context` (`CORE_V2` when none is given), whatever it holds.

    The result holds everything the read finds: the dtype, the compressor
    and each filter as the scope read them (an `AcceptedField` read by
    the definition that claims the typestr or id, an `UnclaimedField`, or
    a `RefusedField`), every problem `validate_array_metadata_v2` finds,
    and, when there are no problems, the document's model.
    """
    scope = scoped(context, CORE_V2)
    reading, members = read_array_v2(value, scope)
    if members is None:
        return reading
    model = ZarrV2ArrayMetadata._of(members.document, scope, reading, members)  # pyright: ignore[reportPrivateUsage]
    return model.reading


class ZarrV2ArrayMetadataUpdate(TypedDict, total=False, extra_items=JSONValue | UNSET):
    """The members `ZarrV2ArrayMetadata.update` replaces: each in the form a document writes it, or `UNSET` to remove a member that a document may omit.

    The members a document may omit are `attributes` (meaning no
    `.zattrs`), `dimension_separator` (read as `"."`), and any member the
    spec does not define.
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
    `UnclaimedField`. `attributes` is `UNSET` when no `.zattrs` exists,
    which is distinct from an empty one. A member the spec does not
    define is kept in `extra_fields`. A model is only built by reading:
    the constructor reads `document` in `context` and raises
    `MetadataValidationError` with every problem, so no model is invalid.
    Two models are equal when their documents mean the same thing in
    their scopes, as the model key defines. `update` reads new members
    in the model's own scope. `with_context` and `refined_in` read the
    document in another scope. A model pickles as its pair.
    """

    __slots__ = ("_claims", "_context", "_document", "_members", "_reading", "_shown")

    zarr_format: Final = 2

    @property
    def claims(self) -> Claims:
        """The claims the reading recorded for each typestr and codec id the document writes, keyed by kind and the name the scope files it under. Computed on first access."""
        try:
            return self._claims
        except AttributeError:
            self._claims = MappingProxyType(claims_of(self._reading.fields()))
            return self._claims

    def __init__(self, document: object, context: ZarrV2Context | None = None) -> None:
        scope = scoped(context, CORE_V2)
        reading, members = read_array_v2(document, scope)
        if members is None:
            raise MetadataValidationError(reading.problems)
        self._adopt(members.document, scope, reading, members)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: Context,
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
        context: Context,
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
            UNSET if members.attributes is UNSET else frozen(members.attributes),
            frozen(members.extra_fields),
        )

    # --- the pair ---------------------------------------------------------

    @property
    def context(self) -> Context:
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
        as text. `attributes` is `UNSET` when there is no `.zattrs`.
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
            UNSET if members.attributes is UNSET else json_text(members.attributes),
            json_text(members.extra_fields),
        )

    def _fill_value_key(self) -> str:
        """The fill value as `==` compares it, as JSON text: in its canonical spelling when a definition in scope read the dtype, and as written otherwise."""
        fill_value = self._members.fill_value
        if isinstance(self.dtype, AcceptedField):
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
        """The separator between chunk indices in a chunk key: `"."` when the document writes none."""
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
        document: dict[str, object] = {**self._document, **members}
        for key, value in members.items():
            if value is UNSET:
                del document[key]
        return type(self)(document, context=self._context)

    def with_context(self, context: ZarrV2Context | None = None) -> ZarrV2ArrayMetadata:
        """Return this document read in `context`, whatever that changes: a gain, a loss, or a conflict.

        Raises `MetadataValidationError` when the document has a problem
        in `context`. The existing reading is reused when `context` reads
        every claim identically.
        """
        scope = scoped(context, CORE_V2)
        if scope.disagreements(self.claims).agrees:
            return self._of(self._document, scope, self._reading, self._members)
        return type(self)(self._document, context=scope)

    def refined_in(self, context: ZarrV2Context | None = None) -> ZarrV2ArrayMetadata:
        """Return this document read in `context`. `context` may claim names this scope left unclaimed, but must not contradict any claim.

        Raises `ScopeConflictError` when `context` reads a typestr or id
        with a different definition, or with no definition, where this
        scope read it with one. The error names each one and its location
        in the document. Raises `MetadataValidationError` when a definition
        that `context` claims rejects what was written.
        """
        scope = scoped(context, CORE_V2)
        found = scope.disagreements(self.claims)
        if len(found.conflicts) != 0:
            raise ScopeConflictError(located_conflicts(self._reading.fields(), found.conflicts))
        return self.with_context(scope)

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
        given: dict[str, object] = dict(overrides)
        if "shape" in given and "chunks" not in given:
            lengths, _ = dimension_lengths(given, "shape")
            if lengths is not None:
                given["chunks"] = lengths
        if "dtype" in given and "fill_value" not in given:
            dtype, _ = resolve_dtype_v2(given["dtype"], context)
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
        valid.
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
