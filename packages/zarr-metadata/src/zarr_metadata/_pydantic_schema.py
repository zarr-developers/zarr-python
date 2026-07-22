"""Private input types used only to generate accurate Pydantic JSON schemas."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003  # resolved by Pydantic at runtime
from typing import Annotated, Literal, NotRequired

from pydantic import ConfigDict, Field, with_config
from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v2.array import (  # noqa: TC001  # resolved by Pydantic at runtime
    ZarrV2DataTypeMetadata,
)
from zarr_metadata.v2.codec import (  # noqa: TC001  # resolved by Pydantic at runtime
    ZarrV2CodecMetadata,
)

NonNegativeInt = Annotated[int, Field(ge=0)]


@with_config(ConfigDict(extra="forbid"))
class ZarrV3NamedConfigJSON(TypedDict):
    """Closed v3 named configuration accepted at optional extension points."""

    name: str
    configuration: NotRequired[Mapping[str, JSONValue]]
    must_understand: NotRequired[bool]


@with_config(ConfigDict(extra="forbid"))
class ZarrV3MandatoryNamedConfigJSON(TypedDict):
    """Closed named configuration accepted where understanding is mandatory."""

    name: str
    configuration: NotRequired[Mapping[str, JSONValue]]
    must_understand: NotRequired[Literal[True]]


ZarrV3MetadataFieldJSON = str | ZarrV3NamedConfigJSON
ZarrV3MandatoryMetadataFieldJSON = str | ZarrV3MandatoryNamedConfigJSON
ZarrV3CodecPipelineJSON = Annotated[tuple[ZarrV3MetadataFieldJSON, ...], Field(min_length=1)]


class ZarrV3ArrayMetadataJSON(TypedDict, extra_items=JSONValue):
    """Schema input for a v3 array document, including arbitrary extensions."""

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: ZarrV3MandatoryMetadataFieldJSON
    shape: tuple[NonNegativeInt, ...]
    chunk_grid: ZarrV3MandatoryMetadataFieldJSON
    chunk_key_encoding: ZarrV3MandatoryMetadataFieldJSON
    fill_value: JSONValue
    codecs: ZarrV3CodecPipelineJSON
    attributes: NotRequired[Mapping[str, JSONValue]]
    storage_transformers: NotRequired[tuple[ZarrV3MetadataFieldJSON, ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]


@with_config(ConfigDict(extra="forbid"))
class ZarrV3ConsolidatedMetadataJSON(TypedDict):
    """Schema input for the closed inline consolidated-metadata envelope."""

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, ZarrV3ArrayMetadataJSON | ZarrV3GroupMetadataJSON]


class ZarrV3GroupMetadataJSON(TypedDict, extra_items=JSONValue):
    """Schema input for a v3 group document, including arbitrary extensions."""

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, JSONValue]]
    consolidated_metadata: NotRequired[ZarrV3ConsolidatedMetadataJSON | None]


@with_config(ConfigDict(extra="forbid"))
class ZarrV2ArrayMetadataJSON(TypedDict):
    """Schema input for the closed, merged v2 array representation."""

    zarr_format: Literal[2]
    shape: tuple[NonNegativeInt, ...]
    chunks: tuple[NonNegativeInt, ...]
    dtype: ZarrV2DataTypeMetadata
    compressor: ZarrV2CodecMetadata | None
    fill_value: JSONValue
    order: Literal["C", "F"]
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    dimension_separator: NotRequired[Literal[".", "/"]]
    attributes: NotRequired[Mapping[str, JSONValue]]


@with_config(ConfigDict(extra="forbid"))
class ZarrV2GroupMetadataJSON(TypedDict):
    """Schema input for the closed, merged v2 group representation."""

    zarr_format: Literal[2]
    attributes: NotRequired[Mapping[str, JSONValue]]


@with_config(ConfigDict(extra="forbid"))
class ZarrV2ConsolidatedMetadataJSON(TypedDict):
    """Schema input matching the v2 consolidated model's structural parser."""

    zarr_consolidated_format: Literal[1]
    metadata: Mapping[str, JSONValue]


__all__ = [
    "ZarrV2ArrayMetadataJSON",
    "ZarrV2ConsolidatedMetadataJSON",
    "ZarrV2GroupMetadataJSON",
    "ZarrV3ArrayMetadataJSON",
    "ZarrV3ConsolidatedMetadataJSON",
    "ZarrV3GroupMetadataJSON",
    "ZarrV3MetadataFieldJSON",
]
