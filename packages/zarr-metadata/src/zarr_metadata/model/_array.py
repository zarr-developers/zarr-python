"""In-memory models for Zarr array metadata documents."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

from typing_extensions import TypedDict, Unpack

from zarr_metadata.model._validation import (
    ARRAY_METADATA_STANDARD_KEYS_V3,
    arrays_to_tuples,
    parse_array_metadata_v2,
    parse_array_metadata_v3,
    parse_metadata_field_v3,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata import (
        ArrayDimensionSeparatorV2,
        ArrayMetadataV2,
        ArrayMetadataV3,
        ArrayOrderV2,
        ExtensionFieldV3,
        MetadataV3,
    )
    from zarr_metadata._common import JSONValue
    from zarr_metadata.v2 import CodecMetadataV2, DataTypeMetadataV2

ArrayMetadataStoreKeyV3 = Literal["zarr.json"]
ARRAY_METADATA_STORE_KEY_V3: Final[ArrayMetadataStoreKeyV3] = "zarr.json"

ArrayMetadataStoreKeyV2 = Literal[".zarray"]
ARRAY_METADATA_STORE_KEY_V2: Final[ArrayMetadataStoreKeyV2] = ".zarray"

AttributesStoreKeyV2 = Literal[".zattrs"]
ATTRIBUTES_STORE_KEY_V2: Final[AttributesStoreKeyV2] = ".zattrs"


@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrMetadataV3:
    """A v3 metadata field in normalized form: a name plus a configuration.

    This is the in-memory model of `MetadataV3` (a bare name string or a
    `{name, configuration}` mapping): the bare-name and missing-configuration
    forms normalize to an empty configuration.
    """

    name: str
    configuration: dict[str, JSONValue]

    def to_json(self) -> MetadataV3:
        return {"name": self.name, "configuration": self.configuration}

    @classmethod
    def from_json(cls, data: object) -> ZarrMetadataV3:
        field = parse_metadata_field_v3(data)
        if isinstance(field, str):
            return cls(name=field, configuration={})
        configuration = arrays_to_tuples(dict(field.get("configuration", {})))
        return cls(name=field["name"], configuration=configuration)  # type: ignore[arg-type]


class ArrayMetadataModelV3Partial(TypedDict, total=False):
    """
    Partial form of the constructor-settable fields of `ArrayMetadataModelV3`.

    Every key is optional and typed with the model's own (not serialized)
    value types, so it describes valid keyword arguments to
    `ArrayMetadataModelV3.update`. The `init=False` fields `zarr_format` and
    `node_type` are intentionally excluded, since they cannot be passed to
    `dataclasses.replace`.

    Drift between this type and the model's settable fields is prevented by
    `tests/model/test_array.py::test_partial_keys_match_settable_model_fields`.
    """

    shape: tuple[int, ...]
    fill_value: JSONValue
    data_type: ZarrMetadataV3
    chunk_grid: ZarrMetadataV3
    codecs: tuple[ZarrMetadataV3, ...]
    chunk_key_encoding: ZarrMetadataV3
    dimension_names: tuple[str | None, ...] | None
    attributes: dict[str, JSONValue]
    storage_transformers: tuple[ZarrMetadataV3, ...]
    extra_fields: dict[str, ExtensionFieldV3]


@dataclass(frozen=True, slots=True, kw_only=True)
class ArrayMetadataModelV3:
    """In-memory model of a v3 array metadata document.

    A canonical, lossless representation of the `zarr.json` content for an
    array. Extension points (`data_type`, `chunk_grid`, `chunk_key_encoding`,
    `codecs`, `storage_transformers`) are held as `ZarrMetadataV3` name +
    configuration pairs and are never interpreted; `fill_value` is held
    verbatim in its JSON form.
    """

    zarr_format: Literal[3] = field(default=3, init=False)
    node_type: Literal["array"] = field(default="array", init=False)
    shape: tuple[int, ...]
    fill_value: JSONValue
    data_type: ZarrMetadataV3
    chunk_grid: ZarrMetadataV3
    codecs: tuple[ZarrMetadataV3, ...]
    chunk_key_encoding: ZarrMetadataV3
    dimension_names: tuple[str | None, ...] | None
    attributes: dict[str, JSONValue]
    storage_transformers: tuple[ZarrMetadataV3, ...]
    extra_fields: dict[str, ExtensionFieldV3]

    @classmethod
    def create_default(
        cls, **overrides: Unpack[ArrayMetadataModelV3Partial]
    ) -> ArrayMetadataModelV3:
        """
        Create a default (empty) v3 array metadata model, with optional overrides.

        The default is a structurally-valid scalar `uint8` array — the array
        analog of `list()` returning `[]`. Any field can be overridden by keyword
        (the same fields accepted by `update`).
        """
        default = cls(
            shape=(),
            fill_value=0,
            data_type=ZarrMetadataV3(name="uint8", configuration={}),
            chunk_grid=ZarrMetadataV3(name="regular", configuration={"chunk_shape": ()}),
            codecs=(ZarrMetadataV3(name="bytes", configuration={}),),
            chunk_key_encoding=ZarrMetadataV3(name="default", configuration={}),
            dimension_names=None,
            attributes={},
            storage_transformers=(),
            extra_fields={},
        )
        return default.update(**overrides)

    def update(self, **kwargs: Unpack[ArrayMetadataModelV3Partial]) -> ArrayMetadataModelV3:
        """
        Return a new `ArrayMetadataModelV3` with the given fields updated.

        Only the constructor-settable fields listed in
        `ArrayMetadataModelV3Partial` can be updated; any attempt to update
        other fields (including the fixed `zarr_format` / `node_type`) is
        rejected at the type level. Each given field fully replaces its
        previous value, including `extra_fields`.

        This is useful for test fixtures that want to override a few fields of a
        base template without having to re-specify the entire document.
        """
        return dataclasses.replace(self, **kwargs)

    def __post_init__(self) -> None:
        if set(self.extra_fields.keys()).intersection(ARRAY_METADATA_STANDARD_KEYS_V3):
            raise ValueError("Extra fields cannot overlap with standard ArrayMetadataV3 fields")

    def to_json(self) -> ArrayMetadataV3:
        out: ArrayMetadataV3 = {
            "zarr_format": self.zarr_format,
            "node_type": self.node_type,
            "shape": self.shape,
            "fill_value": self.fill_value,
            "data_type": self.data_type.to_json(),
            "chunk_grid": self.chunk_grid.to_json(),
            "codecs": tuple(codec.to_json() for codec in self.codecs),
            "chunk_key_encoding": self.chunk_key_encoding.to_json(),
        }
        if self.dimension_names is not None:
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
    def from_json(cls, data: object) -> ArrayMetadataModelV3:
        parsed = parse_array_metadata_v3(arrays_to_tuples(data))
        extra_fields: dict[str, ExtensionFieldV3] = {
            k: v  # type: ignore[misc]
            for k, v in parsed.items()
            if k not in ARRAY_METADATA_STANDARD_KEYS_V3
        }
        return cls(
            shape=parsed["shape"],
            fill_value=parsed["fill_value"],  # type: ignore[arg-type]  # fill_value: object in upstream TypedDict
            data_type=ZarrMetadataV3.from_json(parsed["data_type"]),
            chunk_grid=ZarrMetadataV3.from_json(parsed["chunk_grid"]),
            codecs=tuple(ZarrMetadataV3.from_json(c) for c in parsed["codecs"]),
            chunk_key_encoding=ZarrMetadataV3.from_json(parsed["chunk_key_encoding"]),
            dimension_names=parsed.get("dimension_names"),
            attributes=dict(parsed.get("attributes", {})),
            storage_transformers=tuple(
                ZarrMetadataV3.from_json(t) for t in parsed.get("storage_transformers", ())
            ),
            extra_fields=extra_fields,
        )

    @classmethod
    def from_key_value(cls, mapping: Mapping[str, bytes]) -> ArrayMetadataModelV3:
        return cls.from_json(json.loads(mapping[ARRAY_METADATA_STORE_KEY_V3]))

    def to_key_value(self, *, indent: int | str | None = None) -> Mapping[str, bytes]:
        return {
            ARRAY_METADATA_STORE_KEY_V3: json.dumps(self.to_json(), indent=indent).encode("utf-8")
        }


class ArrayMetadataModelV2Partial(TypedDict, total=False):
    """
    Partial form of the constructor-settable fields of `ArrayMetadataModelV2`.

    Every key is optional and typed with the model's own value types, so it
    describes valid keyword arguments to `ArrayMetadataModelV2.update` and
    `create_default`. The `init=False` field `zarr_format` is intentionally
    excluded, since it cannot be passed to `dataclasses.replace`.

    Drift between this type and the model's settable fields is prevented by
    `tests/model/test_array.py::test_v2_partial_keys_match_settable_model_fields`.
    """

    shape: tuple[int, ...]
    dtype: DataTypeMetadataV2
    chunks: tuple[int, ...]
    fill_value: JSONValue
    order: ArrayOrderV2
    compressor: CodecMetadataV2 | None
    filters: tuple[CodecMetadataV2, ...] | None
    dimension_separator: ArrayDimensionSeparatorV2
    attributes: dict[str, JSONValue]


@dataclass(frozen=True, slots=True, kw_only=True)
class ArrayMetadataModelV2:
    """In-memory model of a v2 array metadata document.

    A canonical, lossless representation of the `.zarray` content plus the
    sibling `.zattrs` attributes. `dtype`, `compressor`, and `filters` are
    held in their raw JSON forms and are never interpreted; `fill_value` is
    held verbatim in its JSON form.
    """

    zarr_format: Literal[2] = field(default=2, init=False)
    shape: tuple[int, ...]
    dtype: DataTypeMetadataV2
    chunks: tuple[int, ...]
    fill_value: JSONValue
    order: ArrayOrderV2
    compressor: CodecMetadataV2 | None
    filters: tuple[CodecMetadataV2, ...] | None
    dimension_separator: ArrayDimensionSeparatorV2 = field(default="/")
    attributes: dict[str, JSONValue]

    def update(self, **kwargs: Unpack[ArrayMetadataModelV2Partial]) -> ArrayMetadataModelV2:
        """
        Return a new `ArrayMetadataModelV2` with the given fields updated.

        Only the constructor-settable fields listed in
        `ArrayMetadataModelV2Partial` can be updated; the fixed `zarr_format` is
        rejected at the type level. Each given field fully replaces its previous
        value.
        """
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def create_default(
        cls, **overrides: Unpack[ArrayMetadataModelV2Partial]
    ) -> ArrayMetadataModelV2:
        """
        Create a default (empty) v2 array metadata model, with optional overrides.

        The default is a structurally-valid scalar `uint8` (`"|u1"`) array — the
        array analog of `list()` returning `[]`. Any field can be overridden by
        keyword (the same fields accepted by `update`).
        """
        default = cls(
            shape=(),
            dtype="|u1",
            chunks=(),
            fill_value=0,
            order="C",
            compressor=None,
            filters=None,
            attributes={},
        )
        return default.update(**overrides)

    def to_json(self) -> ArrayMetadataV2:
        out: ArrayMetadataV2 = {
            "zarr_format": self.zarr_format,
            "shape": self.shape,
            "dtype": self.dtype,
            "order": self.order,
            "chunks": self.chunks,
            "fill_value": self.fill_value,
            "dimension_separator": self.dimension_separator,
            "attributes": self.attributes,
            "compressor": self.compressor,
            "filters": self.filters,
        }
        return out

    @classmethod
    def from_json(cls, data: object) -> ArrayMetadataModelV2:
        parsed = parse_array_metadata_v2(arrays_to_tuples(data))
        return cls(
            shape=parsed["shape"],
            dtype=parsed["dtype"],
            chunks=parsed["chunks"],
            fill_value=parsed["fill_value"],  # type: ignore[arg-type]  # fill_value: object in upstream TypedDict
            order=parsed["order"],
            compressor=parsed["compressor"],
            filters=parsed["filters"],
            dimension_separator=parsed.get("dimension_separator", "/"),
            attributes=dict(parsed.get("attributes", {})),
        )

    @classmethod
    def from_key_value(cls, mapping: Mapping[str, bytes]) -> ArrayMetadataModelV2:
        zarray = json.loads(mapping[ARRAY_METADATA_STORE_KEY_V2])
        zattrs: dict[str, JSONValue] = (
            json.loads(mapping[ATTRIBUTES_STORE_KEY_V2])
            if ATTRIBUTES_STORE_KEY_V2 in mapping
            else {}
        )
        return cls.from_json({**zarray, "attributes": zattrs})

    def to_key_value(self, *, indent: int | str | None = None) -> Mapping[str, bytes]:
        # Attributes live only in the sibling `.zattrs` file; the `.zarray`
        # document must exclude them.
        zarray = {k: v for k, v in self.to_json().items() if k != "attributes"}
        return {
            ARRAY_METADATA_STORE_KEY_V2: json.dumps(zarray, indent=indent).encode("utf-8"),
            ATTRIBUTES_STORE_KEY_V2: json.dumps(self.attributes, indent=indent).encode("utf-8"),
        }
