"""In-memory models for Zarr array metadata documents."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import (
    ARRAY_METADATA_STANDARD_KEYS_V3,
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    load_store_json,
    parse_array_metadata_v2,
    parse_array_metadata_v3,
    parse_metadata_field_v3,
)

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue
    from zarr_metadata.v2.array import (
        ZarrV2ArrayDimensionSeparator,
        ZarrV2ArrayMetadataJSON,
        ZarrV2ArrayOrder,
        ZarrV2DataTypeMetadata,
    )
    from zarr_metadata.v2.codec import ZarrV2CodecMetadata
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON, ZarrV3ExtensionField

ZarrV3ArrayMetadataStoreKey = Literal["zarr.json"]
ARRAY_METADATA_STORE_KEY_V3: Final[ZarrV3ArrayMetadataStoreKey] = "zarr.json"

ZarrV2ArrayMetadataStoreKey = Literal[".zarray"]
ARRAY_METADATA_STORE_KEY_V2: Final[ZarrV2ArrayMetadataStoreKey] = ".zarray"

ZarrV2AttributesStoreKey = Literal[".zattrs"]
ATTRIBUTES_STORE_KEY_V2: Final[ZarrV2AttributesStoreKey] = ".zattrs"


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV3NamedConfig:
    """A v3 metadata field in normalized form: a name plus a configuration.

    This is the in-memory model of `ZarrV3MetadataFieldJSON` (a bare name string or a
    `{name, configuration}` mapping): the bare-name and missing-configuration
    forms normalize to an empty configuration.
    """

    name: str
    configuration: dict[str, JSONValue]

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        return {"name": self.name, "configuration": self.configuration}

    @classmethod
    def from_json(cls, data: object) -> ZarrV3NamedConfig:
        field = parse_metadata_field_v3(data)
        if isinstance(field, str):
            return cls(name=field, configuration={})
        # Sound cast: parse_metadata_field_v3 checked the configuration is a
        # string-keyed mapping of JSON values; arrays_to_tuples only converts
        # lists to tuples within that shape.
        configuration = cast(
            "dict[str, JSONValue]", arrays_to_tuples(dict(field.get("configuration", {})))
        )
        return cls(name=field["name"], configuration=configuration)


ZarrV3MetadataField: TypeAlias = ZarrV3NamedConfig
"""The in-memory model of one field of a v3 metadata document.

This is the role-named alias for annotation positions: model fields and
consumer signatures should say `ZarrV3MetadataField` (the logical meaning)
rather than `ZarrV3NamedConfig` (the serialized form the field currently
takes). Today every metadata field normalizes to a named configuration, so
the alias is exactly `ZarrV3NamedConfig`; if a future spec revision adds a
field form that cannot be normalized to name + configuration, this alias
widens to a union and annotation sites do not change. Mirrors the raw-layer
split between `ZarrV3NamedConfigJSON` (shape) and `ZarrV3MetadataFieldJSON` (field union).
"""


def must_understand_subset(
    extra_fields: Mapping[str, ZarrV3ExtensionField],
) -> dict[str, ZarrV3ExtensionField]:
    """The subset of `extra_fields` the reader is obligated to understand.

    Per the v3 spec, an extension field is implicitly `must_understand: True`
    unless it explicitly says otherwise, and an implementation MUST fail to
    open a group or array carrying fields it does not recognize that are not
    explicitly `must_understand: false`. A non-mapping field value cannot
    carry the explicit waiver, so it always requires understanding (the
    runtime isinstance check defends against values looser than the declared
    `ZarrV3ExtensionField`).
    """
    fields = cast("Mapping[str, object]", extra_fields)
    return cast(
        "dict[str, ZarrV3ExtensionField]",
        {
            name: value
            for name, value in fields.items()
            if not (
                isinstance(value, Mapping)
                and cast("Mapping[str, object]", value).get("must_understand") is False
            )
        },
    )


class ZarrV3ArrayMetadataPartial(TypedDict, total=False):
    """
    Partial form of the constructor-settable fields of `ZarrV3ArrayMetadata`.

    Every key is optional and typed with the model's own (not serialized)
    value types, so it describes valid keyword arguments to
    `ZarrV3ArrayMetadata.update`. The `init=False` fields `zarr_format` and
    `node_type` are intentionally excluded, since they cannot be passed to
    `dataclasses.replace`.

    Drift between this type and the model's settable fields is prevented by
    `tests/model/test_array.py::test_partial_keys_match_settable_model_fields`.
    """

    shape: tuple[int, ...]
    fill_value: JSONValue
    data_type: ZarrV3MetadataField
    chunk_grid: ZarrV3MetadataField
    codecs: tuple[ZarrV3MetadataField, ...]
    chunk_key_encoding: ZarrV3MetadataField
    dimension_names: tuple[str | None, ...] | UNSET
    attributes: dict[str, JSONValue]
    storage_transformers: tuple[ZarrV3MetadataField, ...]
    extra_fields: dict[str, ZarrV3ExtensionField]


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV3ArrayMetadata:
    """In-memory model of a v3 array metadata document.

    A canonical, lossless representation of the `zarr.json` content for an
    array. Extension points (`data_type`, `chunk_grid`, `chunk_key_encoding`,
    `codecs`, `storage_transformers`) are held as `ZarrV3MetadataField`
    values (currently always `ZarrV3NamedConfig` name + configuration pairs)
    and are never interpreted; `fill_value` is held
    verbatim in its JSON form.
    """

    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default="array", init=False)
    shape: tuple[int, ...]
    fill_value: JSONValue
    data_type: ZarrV3MetadataField
    chunk_grid: ZarrV3MetadataField
    codecs: tuple[ZarrV3MetadataField, ...]
    chunk_key_encoding: ZarrV3MetadataField
    dimension_names: tuple[str | None, ...] | UNSET
    attributes: dict[str, JSONValue]
    storage_transformers: tuple[ZarrV3MetadataField, ...]
    extra_fields: dict[str, ZarrV3ExtensionField]

    @classmethod
    def create_default(cls, **overrides: Unpack[ZarrV3ArrayMetadataPartial]) -> ZarrV3ArrayMetadata:
        """
        Create a default (empty) v3 array metadata model, with optional overrides.

        The default is a structurally-valid scalar `uint8` array — the array
        analog of `list()` returning `[]`. Any field can be overridden by keyword
        (the same fields accepted by `update`). Overriding `shape` without
        `chunk_grid` derives a consistent default grid: one regular chunk
        covering the array (`chunk_shape` equal to `shape`).

        The derivation is deliberately one-way. A user-supplied `chunk_grid`
        is an extension point and is taken verbatim — deriving `shape` from
        it would require interpreting the grid's configuration, which this
        layer never does (and cannot do for unrecognized grid names). So
        overriding `chunk_grid` without `shape` keeps the scalar default
        `shape=()`, and consistency between the two is the caller's
        responsibility.
        """
        if "shape" in overrides and "chunk_grid" not in overrides:
            overrides["chunk_grid"] = ZarrV3NamedConfig(
                name="regular", configuration={"chunk_shape": tuple(overrides["shape"])}
            )
        default = cls(
            shape=(),
            fill_value=0,
            data_type=ZarrV3NamedConfig(name="uint8", configuration={}),
            chunk_grid=ZarrV3NamedConfig(name="regular", configuration={"chunk_shape": ()}),
            codecs=(ZarrV3NamedConfig(name="bytes", configuration={}),),
            chunk_key_encoding=ZarrV3NamedConfig(name="default", configuration={}),
            dimension_names=UNSET,
            attributes={},
            storage_transformers=(),
            extra_fields={},
        )
        return default.update(**overrides)

    def update(self, **kwargs: Unpack[ZarrV3ArrayMetadataPartial]) -> ZarrV3ArrayMetadata:
        """
        Return a new `ZarrV3ArrayMetadata` with the given fields updated.

        Only the constructor-settable fields listed in
        `ZarrV3ArrayMetadataPartial` can be updated; any attempt to update
        other fields (including the fixed `zarr_format` / `node_type`) is
        rejected at the type level. Each given field fully replaces its
        previous value, including `extra_fields`.

        This is useful for test fixtures that want to override a few fields of a
        base template without having to re-specify the entire document.

        No re-validation is performed (`update` is `dataclasses.replace`), so
        a repair or edit can produce an invalid document; validity is checked
        on `from_json`, not on field replacement.
        """
        return dataclasses.replace(self, **kwargs)

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

    def to_json(self) -> ZarrV3ArrayMetadataJSON:
        out: ZarrV3ArrayMetadataJSON = {
            "zarr_format": self.zarr_format,
            "node_type": self.node_type,
            "shape": self.shape,
            "fill_value": self.fill_value,
            "data_type": self.data_type.to_json(),
            "chunk_grid": self.chunk_grid.to_json(),
            "codecs": tuple(codec.to_json() for codec in self.codecs),
            "chunk_key_encoding": self.chunk_key_encoding.to_json(),
        }
        if self.dimension_names is not UNSET:
            out["dimension_names"] = self.dimension_names
        if len(self.attributes) > 0:
            out["attributes"] = self.attributes
        if len(self.storage_transformers) > 0:
            out["storage_transformers"] = tuple(
                transformer.to_json() for transformer in self.storage_transformers
            )
        # Extra fields are the TypedDict's `extra_items` (PEP 728). Assign them
        # by key rather than `out.update(**...)`: type checkers understand the
        # indexed-write path against `extra_items`, but not the `update(**...)`
        # overload.
        for key, value in self.extra_fields.items():
            out[key] = value
        return out

    @classmethod
    def from_json(cls, data: object) -> ZarrV3ArrayMetadata:
        parsed = parse_array_metadata_v3(arrays_to_tuples(data))
        # Sound cast: the TypedDict types all non-standard keys as its
        # `extra_items` (`ZarrV3ExtensionField`); the comprehension's inferred value
        # type is the union over ALL keys because the key filter cannot narrow it.
        extra_fields = cast(
            "dict[str, ZarrV3ExtensionField]",
            {k: v for k, v in parsed.items() if k not in ARRAY_METADATA_STANDARD_KEYS_V3},
        )
        return cls(
            shape=parsed["shape"],
            fill_value=parsed["fill_value"],
            data_type=ZarrV3NamedConfig.from_json(parsed["data_type"]),
            chunk_grid=ZarrV3NamedConfig.from_json(parsed["chunk_grid"]),
            codecs=tuple(ZarrV3NamedConfig.from_json(c) for c in parsed["codecs"]),
            chunk_key_encoding=ZarrV3NamedConfig.from_json(parsed["chunk_key_encoding"]),
            dimension_names=parsed.get("dimension_names", UNSET),
            attributes=dict(parsed.get("attributes", {})),
            storage_transformers=tuple(
                ZarrV3NamedConfig.from_json(t) for t in parsed.get("storage_transformers", ())
            ),
            extra_fields=extra_fields,
        )

    @property
    def must_understand_fields(self) -> dict[str, ZarrV3ExtensionField]:
        """Extra fields the reader is obligated to understand.

        Everything in `extra_fields` not explicitly waived with
        `must_understand: false` (the spec's implicit-true rule). A compliant
        reader MUST fail to open the array if this contains any field it does
        not recognize; the model layer only partitions by obligation, since
        recognition is reader-specific.
        """
        return must_understand_subset(self.extra_fields)

    @classmethod
    def from_key_value(cls, mapping: Mapping[str, bytes]) -> ZarrV3ArrayMetadata:
        return cls.from_json(load_store_json(mapping, ARRAY_METADATA_STORE_KEY_V3))

    def to_key_value(self, *, indent: int | str | None = None) -> Mapping[str, bytes]:
        return {
            ARRAY_METADATA_STORE_KEY_V3: json.dumps(self.to_json(), indent=indent).encode("utf-8")
        }


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
        `to_key_value` to produce the spec-conforming split for storage.
        """
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
        return out

    @classmethod
    def from_json(cls, data: object) -> ZarrV2ArrayMetadata:
        parsed = parse_array_metadata_v2(arrays_to_tuples(data))
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
    def from_key_value(cls, mapping: Mapping[str, bytes]) -> ZarrV2ArrayMetadata:
        zarray = load_store_json(mapping, ARRAY_METADATA_STORE_KEY_V2)
        if ATTRIBUTES_STORE_KEY_V2 in mapping:
            zattrs = load_store_json(mapping, ATTRIBUTES_STORE_KEY_V2)
            return cls.from_json({**zarray, "attributes": zattrs})
        return cls.from_json(dict(zarray))

    def to_key_value(self, *, indent: int | str | None = None) -> Mapping[str, bytes]:
        # Attributes live only in the sibling `.zattrs` file; the `.zarray`
        # document must exclude them. The `.zattrs` key is present exactly
        # when attributes are set (even empty) — UNSET emits no file.
        zarray = {k: v for k, v in self.to_json().items() if k != "attributes"}
        out = {ARRAY_METADATA_STORE_KEY_V2: json.dumps(zarray, indent=indent).encode("utf-8")}
        if self.attributes is not UNSET:
            out[ATTRIBUTES_STORE_KEY_V2] = json.dumps(self.attributes, indent=indent).encode(
                "utf-8"
            )
        return out
