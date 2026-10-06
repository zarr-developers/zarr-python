"""In-memory models for Zarr array metadata documents."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    copied,
    json_text,
    refine_json,
    with_input,
)
from zarr_metadata._sentinel import UNSET
from zarr_metadata.model._validation import (
    ArrayMembersV3,
    StoreKey,
    ZarrV3ArrayMetadataReading,
    construct,
    dimension_lengths,
    dump_store_json,
    load_store_json,
    parse_array_metadata_v2,
    read_array_v3,
)
from zarr_metadata.v2.array import ZARR_V2_ARRAY_METADATA_STORE_KEY
from zarr_metadata.v2.attributes import ZARR_V2_ATTRIBUTES_STORE_KEY
from zarr_metadata.v3._definition import (
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    Read,
    StorageTransformerDefinition,
    Unclaimed,
    field_key,
    spelled_canonically,
)
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context
from zarr_metadata.v3._scope import Claims, claims_of
from zarr_metadata.v3.array import ZARR_V3_ARRAY_METADATA_STORE_KEY, ZarrV3ExtensionField

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue
    from zarr_metadata.v2.array import (
        ZarrV2ArrayDimensionSeparator,
        ZarrV2ArrayMetadataJSON,
        ZarrV2ArrayMetadataStoreKey,
        ZarrV2ArrayOrder,
        ZarrV2DataTypeMetadata,
    )
    from zarr_metadata.v2.attributes import ZarrV2AttributesStoreKey
    from zarr_metadata.v2.codec import ZarrV2CodecMetadata
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
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


class ZarrV3ArrayMetadata:
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
    scopes, as `array_key` says; the scope itself takes no part. `update`
    reads new members in the model's own scope; `with_context` and
    `refined_in` read the document in another. A model pickles as its
    pair, when the definitions its scope holds do: ones whose functions
    are defined at a module's top level.
    """

    __slots__ = ("_claims", "_context", "_document", "_key", "_members", "_reading")

    zarr_format: Final = 3
    node_type: Final = "array"

    def __init__(self, document: object, context: Context | None = None) -> None:
        scope = CORE_AND_EXTENSIONS if context is None else context
        reading, members = read_array_v3(document, scope)
        if members is None:
            raise MetadataValidationError(reading.problems)
        refined, _ = refine_json(document)
        self._adopt(cast("dict[str, JSONValue]", refined), scope, reading, members)

    @classmethod
    def _of(
        cls,
        document: dict[str, JSONValue],
        context: Context,
        reading: ZarrV3ArrayMetadataReading,
        members: ArrayMembersV3,
    ) -> ZarrV3ArrayMetadata:
        """A model of a document a read found nothing wrong with, holding that reading: no second read."""
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
        self._reading = reading
        self._members = members
        self._key = array_key(self)
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

    @property
    def claims(self) -> Claims:
        """What the reading claimed of each name the document writes, keyed as the scope files it."""
        return self._claims

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

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._key == cast("ZarrV3ArrayMetadata", other)._key

    def __hash__(self) -> int:
        return hash(self._key)

    # --- typed views ------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """The array's shape."""
        return self._members.shape

    @property
    def fill_value(self) -> JSONValue:
        """The fill value as written."""
        return self._members.fill_value

    @property
    def dimension_names(self) -> tuple[str | None, ...] | UNSET:
        """The dimension names; `UNSET` when the document writes none."""
        return self._members.dimension_names

    @property
    def attributes(self) -> Mapping[str, JSONValue]:
        """The attributes, a read-only view; empty when the document writes none."""
        return MappingProxyType(self._members.attributes)

    @property
    def extra_fields(self) -> Mapping[str, ZarrV3ExtensionField]:
        """Each member the spec does not define, by name, a read-only view."""
        return MappingProxyType(self._members.extra_fields)

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


def array_key(model: ZarrV3ArrayMetadata) -> tuple[object, ...]:
    """What `==` and `hash` compare of a v3 array model: what its document means.

    Each field by its `field_key`, the fill value in its canonical spelling
    as JSON text when a definition in scope read the data type, and every
    other member as it is, the JSON ones as text.
    """
    return (
        model.shape,
        _fill_value_key(model),
        field_key(model.data_type),
        field_key(model.chunk_grid),
        tuple(field_key(codec) for codec in model.codecs),
        field_key(model.chunk_key_encoding),
        model.dimension_names,
        json_text(dict(model.attributes)),
        tuple(field_key(transformer) for transformer in model.storage_transformers),
        json_text(dict(model.extra_fields)),
    )


def _fill_value_key(model: ZarrV3ArrayMetadata) -> str:
    """What `==` compares of `model`'s fill value: its canonical spelling as JSON text when a definition in scope read the data type, and the fill value as written when none did."""
    if isinstance(model.data_type, Read):
        return json_text(spelled_canonically(model.data_type, model.fill_value))
    return json_text(model.fill_value)


def array_json(model: ZarrV3ArrayMetadata) -> ZarrV3ArrayMetadataJSON:
    """`model`'s document, holding the model's own values: what a writer serializes without copying."""
    return cast("ZarrV3ArrayMetadataJSON", model._document)  # pyright: ignore[reportPrivateUsage]


def array_model(
    reading: ZarrV3ArrayMetadataReading, members: ArrayMembersV3
) -> ZarrV3ArrayMetadata:
    """The model of a document its reading found nothing wrong with, its document rebuilt from the reading.

    A shim for the group reader until it hands the nested documents down
    with their readings: the document is rebuilt from what was read, so a
    member written empty is written back as the read holds it.
    """
    document: dict[str, JSONValue] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": members.shape,
        "data_type": cast("Read[Any] | Unclaimed", reading.data_type).json,
        "chunk_grid": cast("Read[Any] | Unclaimed", reading.chunk_grid).json,
        "chunk_key_encoding": cast("Read[Any] | Unclaimed", reading.chunk_key_encoding).json,
        "fill_value": members.fill_value,
        "codecs": tuple(
            cast("Read[Any] | Unclaimed", stage.codec).json for stage in reading.pipeline
        ),
    }
    if members.dimension_names is not UNSET:
        document["dimension_names"] = members.dimension_names
    if len(members.attributes) != 0:
        document["attributes"] = members.attributes
    if len(reading.storage_transformers) != 0:
        document["storage_transformers"] = tuple(
            cast("Read[Any] | Unclaimed", transformer).json
            for transformer in reading.storage_transformers
        )
    document.update(members.extra_fields)
    return ZarrV3ArrayMetadata._of(  # pyright: ignore[reportPrivateUsage]
        document, CORE_AND_EXTENSIONS, reading, members
    )


def read_array_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
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
    reading, members = read_array_v3(value, context)
    if members is None:
        return reading
    refined, _ = refine_json(value)
    document = cast("dict[str, JSONValue]", refined)
    model = ZarrV3ArrayMetadata._of(document, context, reading, members)  # pyright: ignore[reportPrivateUsage]
    return dataclasses.replace(reading, metadata=model)


class ZarrV2ArrayMetadataPartial(TypedDict, total=False):
    """
    Partial form of the constructor-settable fields of `ZarrV2ArrayMetadata`.

    Every key is optional and typed with the model's own value types, so it
    describes valid keyword arguments to `ZarrV2ArrayMetadata.update` and
    `create_default`. The `init=False` field `zarr_format` is intentionally
    excluded, since it cannot be passed to `dataclasses.replace`.

    Drift between this type and the model's settable fields is prevented by
    `tests/model/test_array.py::test_v2_partial_keys_match_settable_model_fields`.
    """

    shape: tuple[int, ...]
    dtype: ZarrV2DataTypeMetadata
    chunks: tuple[int, ...]
    fill_value: JSONValue
    order: ZarrV2ArrayOrder
    compressor: ZarrV2CodecMetadata | None
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    dimension_separator: ZarrV2ArrayDimensionSeparator
    attributes: dict[str, JSONValue] | UNSET


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV2ArrayMetadata:
    """In-memory model of a v2 array metadata document.

    A canonical, lossless representation of the `.zarray` content plus the
    sibling `.zattrs` attributes. `dtype`, `compressor`, and `filters` are
    held in their raw JSON forms and are never interpreted; `fill_value` is
    held verbatim in its JSON form. `attributes` is `UNSET` when no
    `.zattrs` file (or merged `attributes` key) exists — distinct from an
    explicit empty `.zattrs`, which is `{}` and round-trips as a file. One
    spelling normalization: a `.zarray` that omits `dimension_separator`
    means `"."` by the v2 convention, and the model holds and re-emits that
    value explicitly. A model checks itself when it is built, as the v3
    models do: its document has no problem `validate_array_metadata_v2`
    finds, or the constructor raises `MetadataValidationError`, so
    `update` refuses a change that would make one.
    """

    zarr_format: Literal[2] = field(default=2, init=False)
    shape: tuple[int, ...]
    dtype: ZarrV2DataTypeMetadata
    chunks: tuple[int, ...]
    fill_value: JSONValue
    order: ZarrV2ArrayOrder
    compressor: ZarrV2CodecMetadata | None
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    # "." is the v2 convention's default for an ABSENT dimension_separator key;
    # from_json normalizes absence to it (a semantics-preserving spelling
    # normalization, like the v3 bare-string metadata-field form). The value
    # is never None: the document grammar has no null spelling for this field.
    dimension_separator: ZarrV2ArrayDimensionSeparator = field(default=".")
    attributes: dict[str, JSONValue] | UNSET

    def __post_init__(self) -> None:
        # Held as a read refines them, in containers of its own.
        members = _v2_array_members(parse_array_metadata_v2(self.to_json()))
        for name, value in members.items():
            object.__setattr__(self, name, value)

    def update(self, **kwargs: Unpack[ZarrV2ArrayMetadataPartial]) -> ZarrV2ArrayMetadata:
        """
        Return a new `ZarrV2ArrayMetadata` with the given fields updated.

        Only the constructor-settable fields listed in
        `ZarrV2ArrayMetadataPartial` can be updated; the fixed `zarr_format` is
        rejected at the type level. Each given field fully replaces its previous
        value. `MetadataValidationError` when the document the change makes
        has a problem, as the model checks itself when it is built.
        """
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def create_default(cls, **overrides: Unpack[ZarrV2ArrayMetadataPartial]) -> ZarrV2ArrayMetadata:
        """
        Create a default (empty) v2 array metadata model, with optional overrides.

        The default is a structurally-valid scalar `uint8` (`"|u1"`) array — the
        array analog of `list()` returning `[]`. Any field can be overridden by
        keyword (the same fields accepted by `update`). Overriding `shape`
        without `chunks` derives `chunks` equal to `shape` (one chunk covering
        the array).

        The derivation is deliberately one-way, matching the v3 model:
        overriding `chunks` without `shape` keeps the scalar default
        `shape=()`, which `chunks` of any other rank do not fit, so
        `MetadataValidationError`, as the v3 model refuses a grid its
        default shape does not take.
        """
        if "shape" in overrides and "chunks" not in overrides:
            overrides["chunks"] = tuple(overrides["shape"])
        default = cls(
            shape=(),
            dtype="|u1",
            chunks=(),
            fill_value=0,
            order="C",
            compressor=None,
            filters=None,
            attributes=UNSET,
        )
        return default.update(**overrides)

    def __eq__(self, other: object) -> bool:
        """Whether `other` models the same array: the same document, as JSON text.

        Nothing in a v2 document is interpreted, so two models are one when
        their documents are written alike, which tells `0` from `0.0` and
        `-0.0`, and takes `NaN` for itself. Equal models hash alike.
        """
        if type(other) is not type(self):
            return NotImplemented
        return json_text(self.to_json()) == json_text(cast("ZarrV2ArrayMetadata", other).to_json())

    def __hash__(self) -> int:
        return hash(json_text(self.to_json()))

    def to_json(self) -> ZarrV2ArrayMetadataJSON:
        """Return the merged in-memory document form.

        `attributes` is included when set (even empty). This is not the
        on-disk `.zarray` content: a conforming `.zarray` must exclude
        `attributes` (they live in the sibling `.zattrs` file). Use
        `to_key_value` to produce the spec-conforming split for storage
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330).
        """
        # to_json output shares no mutable state with the model: the document
        # is copied whole, one frame for each level of nesting.
        out: ZarrV2ArrayMetadataJSON = {
            "zarr_format": self.zarr_format,
            "shape": self.shape,
            "dtype": self.dtype,
            "order": self.order,
            "chunks": self.chunks,
            "fill_value": self.fill_value,
            "dimension_separator": self.dimension_separator,
            "compressor": self.compressor,
            "filters": self.filters,
        }
        if self.attributes is not UNSET:
            out["attributes"] = self.attributes
        return cast("ZarrV2ArrayMetadataJSON", copied(cast("JSONValue", out)))

    @classmethod
    def from_json(cls, data: object) -> ZarrV2ArrayMetadata:
        """The model of `data`, a v2 array document with its attributes under `attributes`.

        `MetadataValidationError` with every problem `validate_array_metadata_v2`
        finds. A missing `dimension_separator` is read as `"."`, which is
        written back. The model shares no mutable state with `data`.
        """
        # A read model shares no mutable state with what it read.
        parsed = cast(
            "ZarrV2ArrayMetadataJSON", copied(cast("JSONValue", parse_array_metadata_v2(data)))
        )
        return construct(cls, **_v2_array_members(parsed))

    @classmethod
    def from_key_value(cls, mapping: Mapping[StoreKey, bytes]) -> ZarrV2ArrayMetadata:
        """The model of the array at `.zarray` in `mapping`, with the attributes at `.zattrs` when there is one.

        `MetadataValidationError` when `.zarray` is missing, bytes are not
        JSON, `.zarray` holds `attributes`, or the document is not valid.
        """
        zarray_raw = load_store_json(mapping, ZARR_V2_ARRAY_METADATA_STORE_KEY)
        if not isinstance(zarray_raw, Mapping):
            return cls.from_json(zarray_raw)
        zarray = cast("Mapping[str, object]", zarray_raw)
        if "attributes" in zarray:
            refused = ValidationProblem(
                ("attributes",), "unexpected document member", "invalid_value"
            )
            raise MetadataValidationError(with_input((refused,), zarray))
        if ZARR_V2_ATTRIBUTES_STORE_KEY in mapping:
            zattrs = load_store_json(mapping, ZARR_V2_ATTRIBUTES_STORE_KEY)
            return cls.from_json({**zarray, "attributes": zattrs})
        return cls.from_json(zarray)

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV2ArrayMetadataStoreKey | ZarrV2AttributesStoreKey, bytes]:
        """The document as a store holds it: `.zarray` without the attributes, and `.zattrs` with them when they are set, even empty.

        A model was checked when it was built, so its document is written
        as it is.
        """
        # Attributes live only in the sibling `.zattrs` file; the `.zarray`
        # document must exclude them. The `.zattrs` key is present exactly
        # when attributes are set (even empty) — UNSET emits no file.
        document = self.to_json()
        zarray = {k: v for k, v in document.items() if k != "attributes"}
        out: dict[ZarrV2ArrayMetadataStoreKey | ZarrV2AttributesStoreKey, bytes] = {
            ZARR_V2_ARRAY_METADATA_STORE_KEY: dump_store_json(zarray, indent=indent)
        }
        if "attributes" in document:
            out[ZARR_V2_ATTRIBUTES_STORE_KEY] = dump_store_json(
                document["attributes"], indent=indent
            )
        return out


def _v2_array_members(document: ZarrV2ArrayMetadataJSON) -> dict[str, object]:
    """The members of the v2 array model of `document`, which `parse_array_metadata_v2` gave: a missing `dimension_separator` is `"."`."""
    return {
        "shape": document["shape"],
        "dtype": document["dtype"],
        "chunks": document["chunks"],
        "fill_value": document["fill_value"],
        "order": document["order"],
        "compressor": document["compressor"],
        "filters": document["filters"],
        "dimension_separator": document.get("dimension_separator", "."),
        "attributes": dict(document["attributes"]) if "attributes" in document else UNSET,
    }
