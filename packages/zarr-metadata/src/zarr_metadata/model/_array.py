"""In-memory models for Zarr array metadata documents."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata._json import (
    MetadataValidationError,
    ValidationProblem,
    copied,
)
from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import (
    ARRAY_METADATA_STANDARD_KEYS_V3,
    NO_SCOPE,
    ArrayMembersV3,
    StoreKey,
    ZarrV3ArrayMetadataReading,
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
    document_json,
)
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context
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


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV3ArrayMetadata:
    """In-memory model of a v3 array metadata document.

    A canonical, semantically lossless representation of the `zarr.json`
    content for an array. Each extension point -- `data_type`,
    `chunk_grid`, `chunk_key_encoding`, each codec and storage transformer
    -- is held as a scope read it: `Read`, holding the definition that
    read it, or `Unclaimed`, an extension that scope left unjudged.
    `fill_value` is held verbatim in its JSON form.

    A model holds no scope: each field keeps the definition that read it,
    and a scope is asked only to read new JSON -- by `from_json`,
    `create_default`, and `update`, which each take one. `to_key_value`
    reads the document it writes by the model's own fields, and refuses
    one with a problem, so a model changed by hand, as
    `dataclasses.replace` changes one, is never written invalid. `to_json`
    writes each extension point as its readers take it, as `Read.to_json`
    says. A model pickles when the definitions its fields hold do: ones
    whose functions are defined at a module's top level.
    """

    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default="array", init=False)
    shape: tuple[int, ...]
    fill_value: JSONValue
    data_type: Read[DataTypeDefinition[Any]] | Unclaimed
    chunk_grid: Read[ChunkGridDefinition[Any]] | Unclaimed
    codecs: tuple[Read[CodecDefinition[Any]] | Unclaimed, ...]
    chunk_key_encoding: Read[ChunkKeyEncodingDefinition[Any]] | Unclaimed
    dimension_names: tuple[str | None, ...] | UNSET
    attributes: dict[str, JSONValue]
    storage_transformers: tuple[Read[StorageTransformerDefinition[Any]] | Unclaimed, ...]
    extra_fields: dict[str, ZarrV3ExtensionField]

    @classmethod
    def create_default(
        cls,
        *,
        context: Context = CORE_AND_EXTENSIONS,
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
        dimension of length 0, which every reader takes: the core spec
        allows 0 there and the regular grid spec does not,
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
        return cls.from_json({**document, **overrides}, context=context)

    def update(
        self, *, context: Context, **members: Unpack[ZarrV3ArrayMetadataUpdate]
    ) -> ZarrV3ArrayMetadata:
        """This model with `members` in their place, each read in `context`; `UNSET` leaves an optional member out.

        Only the members given are read in `context`: each field the model
        holds is kept as it was read, whatever scope read it. The document
        they make is then read as a whole, so `MetadataValidationError`
        when it has a problem, and members that go together are passed
        together: a `shape` with a grid that fits it.
        """
        document = {**held_document(self), **members}
        for key, value in members.items():
            if value is UNSET:
                del document[key]
        return type(self).from_json(document, context=context)

    def __post_init__(self) -> None:
        overlap = set(self.extra_fields.keys()).intersection(ARRAY_METADATA_STANDARD_KEYS_V3)
        if overlap:
            raise MetadataValidationError(
                [
                    ValidationProblem(
                        ("extra_fields",),
                        "Extra fields cannot overlap with standard Zarr V3 array metadata fields",
                        "invalid_value",
                    )
                ]
            )
        # The runtime half of the annotations: each extension point a field
        # read as its kind, read or unclaimed, as a read gives it.
        for key, kind, nodes in (
            ("data_type", DataTypeDefinition, (self.data_type,)),
            ("chunk_grid", ChunkGridDefinition, (self.chunk_grid,)),
            ("chunk_key_encoding", ChunkKeyEncodingDefinition, (self.chunk_key_encoding,)),
            ("codecs", CodecDefinition, self.codecs),
            ("storage_transformers", StorageTransformerDefinition, self.storage_transformers),
        ):
            for node in cast("tuple[object, ...]", nodes):
                if not isinstance(node, (Read, Unclaimed)) or node.read_as is not kind:
                    msg = f"{key}: expected a field read as a {kind.__name__}, got {node!r}"
                    raise TypeError(msg)

    def to_json(self) -> ZarrV3ArrayMetadataJSON:
        """The document as JSON, arrays as tuples, sharing no mutable state with the model.

        Each extension point as its readers take it, as `Read.to_json`
        writes one; `dimension_names` when set, and `attributes` and
        `storage_transformers` when not empty.
        """
        return cast("ZarrV3ArrayMetadataJSON", copied(cast("JSONValue", array_json(self))))

    @classmethod
    def from_json(
        cls, data: object, *, context: Context = CORE_AND_EXTENSIONS
    ) -> ZarrV3ArrayMetadata:
        """The model of `data`, a v3 array document read in `context`.

        `MetadataValidationError` with every problem the read finds.
        `read_array_metadata_v3` gives the reading this model is built
        from, and the problems of a document with some.
        """
        reading = read_array_metadata_v3(data, context=context)
        if reading.metadata is None:
            raise MetadataValidationError(reading.problems)
        return reading.metadata

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

    @classmethod
    def from_key_value(
        cls, mapping: Mapping[StoreKey, bytes], *, context: Context = CORE_AND_EXTENSIONS
    ) -> ZarrV3ArrayMetadata:
        """The model of the array document at `zarr.json` in `mapping`, read in `context`.

        `MetadataValidationError` when the key is missing, its bytes are not
        JSON, or the document is not valid.
        """
        return cls.from_json(
            load_store_json(mapping, ZARR_V3_ARRAY_METADATA_STORE_KEY), context=context
        )

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV3ArrayMetadataStoreKey, bytes]:
        """The document as a store holds it: JSON bytes at `zarr.json`, indented by `indent`.

        `MetadataValidationError` when the document has a problem, as the
        model's own fields read it, so no invalid document is written, even
        of a model changed by hand. `NaN`, `Infinity` and `-Infinity` in
        `attributes` are written as those bare tokens, as zarr-python
        writes them, which a strict JSON parser refuses.
        """
        problems = array_problems(self)
        if len(problems) != 0:
            raise MetadataValidationError(problems)
        return {ZARR_V3_ARRAY_METADATA_STORE_KEY: dump_store_json(array_json(self), indent=indent)}


def array_problems(model: ZarrV3ArrayMetadata) -> tuple[ValidationProblem, ...]:
    """What is wrong with `model`'s document, as the model's own fields read it: nothing, for a model a read built."""
    return read_array_v3(held_document(model), NO_SCOPE)[0].problems


def array_json(model: ZarrV3ArrayMetadata) -> ZarrV3ArrayMetadataJSON:
    """`model`'s document as JSON, holding the model's own values: what `to_key_value` serializes, which changes nothing, and `to_json` copies."""
    return cast("ZarrV3ArrayMetadataJSON", _document(model, document_json))


def held_document(model: ZarrV3ArrayMetadata) -> dict[str, object]:
    """`model`'s document with each field as it was read, which a read takes as it is: what `update` and `to_key_value` read, reading no field again."""
    return _document(model, _as_read)


def _as_read(field: Read[Any] | Unclaimed) -> object:
    return field


def _document(
    model: ZarrV3ArrayMetadata, write: Callable[[Read[Any] | Unclaimed], object]
) -> dict[str, object]:
    """`model`'s document, each field as `write` gives it, the rest as the model holds it."""
    out: dict[str, object] = {
        "zarr_format": model.zarr_format,
        "node_type": model.node_type,
        "shape": model.shape,
        "fill_value": model.fill_value,
        "data_type": write(model.data_type),
        "chunk_grid": write(model.chunk_grid),
        "codecs": tuple(write(codec) for codec in model.codecs),
        "chunk_key_encoding": write(model.chunk_key_encoding),
    }
    if model.dimension_names is not UNSET:
        out["dimension_names"] = model.dimension_names
    if len(model.attributes) > 0:
        out["attributes"] = model.attributes
    if len(model.storage_transformers) > 0:
        out["storage_transformers"] = tuple(
            write(transformer) for transformer in model.storage_transformers
        )
    out.update(model.extra_fields)
    return out


def read_array_metadata_v3(
    value: object, *, context: Context = CORE_AND_EXTENSIONS
) -> ZarrV3ArrayMetadataReading:
    """`value`, a v3 array document, as `context` read it, whatever it holds.

    Everything a read finds, in one: each extension point as `context`
    read it -- `Read` by the definition that claims its name, `Unclaimed`,
    or `Refused` -- the chunks the codecs are handed, each codec with the
    chunk it is handed, every problem `validate_array_metadata_v3` finds,
    and, when there is none, the document's model, holding the same
    fields. A policy over the fields, the core spec's alone, say, is a
    walk over its `fields()`. A value that is not an object holds no
    field.
    """
    reading, members = read_array_v3(value, context)
    if members is None:
        return reading
    return dataclasses.replace(reading, metadata=array_model(reading, members))


def array_model(
    reading: ZarrV3ArrayMetadataReading, members: ArrayMembersV3
) -> ZarrV3ArrayMetadata:
    """The model of a document its reading found nothing wrong with: its fields as read, and its other members as the read refined them."""
    return ZarrV3ArrayMetadata(
        shape=members.shape,
        fill_value=members.fill_value,
        data_type=cast("Read[DataTypeDefinition[Any]] | Unclaimed", reading.data_type),
        chunk_grid=cast("Read[ChunkGridDefinition[Any]] | Unclaimed", reading.chunk_grid),
        codecs=tuple(
            cast("Read[CodecDefinition[Any]] | Unclaimed", stage.codec)
            for stage in reading.pipeline
        ),
        chunk_key_encoding=cast(
            "Read[ChunkKeyEncodingDefinition[Any]] | Unclaimed", reading.chunk_key_encoding
        ),
        dimension_names=members.dimension_names,
        attributes=members.attributes,
        storage_transformers=cast(
            "tuple[Read[StorageTransformerDefinition[Any]] | Unclaimed, ...]",
            reading.storage_transformers,
        ),
        extra_fields=members.extra_fields,
    )


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
    value explicitly.
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

    def update(self, **kwargs: Unpack[ZarrV2ArrayMetadataPartial]) -> ZarrV2ArrayMetadata:
        """
        Return a new `ZarrV2ArrayMetadata` with the given fields updated.

        Only the constructor-settable fields listed in
        `ZarrV2ArrayMetadataPartial` can be updated; the fixed `zarr_format` is
        rejected at the type level. Each given field fully replaces its previous
        value.
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
        `shape=()`, and consistency between the two is the caller's
        responsibility.
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

    def to_json(self) -> ZarrV2ArrayMetadataJSON:
        """Return the merged in-memory document form.

        `attributes` is included when set (even empty). This is not the
        on-disk `.zarray` content: a conforming `.zarray` must exclude
        `attributes` (they live in the sibling `.zattrs` file). Use
        `to_key_value` to produce the spec-conforming split for storage
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330).
        """
        # to_json output shares no mutable state with the model: every value
        # that can hold a mutable container is deep-copied.
        out: ZarrV2ArrayMetadataJSON = {
            "zarr_format": self.zarr_format,
            "shape": self.shape,
            "dtype": self.dtype,
            "order": self.order,
            "chunks": self.chunks,
            "fill_value": copy.deepcopy(self.fill_value),
            "dimension_separator": self.dimension_separator,
            "compressor": copy.deepcopy(self.compressor),
            "filters": copy.deepcopy(self.filters),
        }
        if self.attributes is not UNSET:
            out["attributes"] = copy.deepcopy(self.attributes)
        return out

    @classmethod
    def from_json(cls, data: object) -> ZarrV2ArrayMetadata:
        """The model of `data`, a v2 array document with its attributes under `attributes`.

        `MetadataValidationError` with every problem `validate_array_metadata_v2`
        finds. A missing `dimension_separator` is read as `"."`, which is
        written back. The model shares no mutable state with `data`.
        """
        # A read model shares no mutable state with what it read.
        parsed = copy.deepcopy(parse_array_metadata_v2(data))
        return cls(
            shape=parsed["shape"],
            dtype=parsed["dtype"],
            chunks=parsed["chunks"],
            fill_value=parsed["fill_value"],
            order=parsed["order"],
            compressor=parsed["compressor"],
            filters=parsed["filters"],
            dimension_separator=parsed.get("dimension_separator", "."),
            attributes=(dict(parsed["attributes"]) if "attributes" in parsed else UNSET),
        )

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
            raise MetadataValidationError(
                [
                    ValidationProblem(
                        ("attributes",),
                        "unexpected document member",
                        "invalid_value",
                    )
                ]
            )
        if ZARR_V2_ATTRIBUTES_STORE_KEY in mapping:
            zattrs = load_store_json(mapping, ZARR_V2_ATTRIBUTES_STORE_KEY)
            return cls.from_json({**zarray, "attributes": zattrs})
        return cls.from_json(zarray)

    def to_key_value(
        self, *, indent: int | str | None = None
    ) -> Mapping[ZarrV2ArrayMetadataStoreKey | ZarrV2AttributesStoreKey, bytes]:
        """The document as a store holds it: `.zarray` without the attributes, and `.zattrs` with them when they are set, even empty.

        Validated first: a model that is not valid raises
        `MetadataValidationError` with every problem, and nothing is written.
        """
        # Attributes live only in the sibling `.zattrs` file; the `.zarray`
        # document must exclude them. The `.zattrs` key is present exactly
        # when attributes are set (even empty) — UNSET emits no file. A model
        # built by hand is not validated: its document is written only if it
        # reads as `from_json` reads one, and every problem is raised.
        document = parse_array_metadata_v2(self.to_json())
        zarray = {k: v for k, v in document.items() if k != "attributes"}
        out: dict[ZarrV2ArrayMetadataStoreKey | ZarrV2AttributesStoreKey, bytes] = {
            ZARR_V2_ARRAY_METADATA_STORE_KEY: dump_store_json(zarray, indent=indent)
        }
        if "attributes" in document:
            out[ZARR_V2_ATTRIBUTES_STORE_KEY] = dump_store_json(
                document["attributes"], indent=indent
            )
        return out
