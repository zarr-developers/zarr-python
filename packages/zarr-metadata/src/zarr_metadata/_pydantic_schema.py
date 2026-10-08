"""Private input types used only to generate accurate Pydantic JSON schemas."""

from __future__ import annotations

import re
from collections.abc import Mapping  # resolved by Pydantic at runtime
from typing import Annotated, Literal, NotRequired

from pydantic import Field
from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v2.array import (  # resolved by Pydantic at runtime
    ZarrV2DataTypeMetadata,
)
from zarr_metadata.v2.codec import (  # resolved by Pydantic at runtime
    ZarrV2CodecMetadata,
)
from zarr_metadata.v3._common import EXTENSION_NAME_SCHEMA_PATTERN

NonNegativeInt = Annotated[int, Field(ge=0)]
ExtensionName = Annotated[str, Field(pattern=re.compile(EXTENSION_NAME_SCHEMA_PATTERN))]
"""A name as the spec names an extension, the pattern `well_named` accepts, so the schema pydantic generates refuses what the reader refuses.

Compiled, so pydantic reads it with Python's `re`, which has the look-ahead
the pattern ends in where its own engine has none, and writes it into the
schema as it is.
"""


class ZarrV3NamedConfigJSON(TypedDict, closed=True):
    """Closed v3 named configuration read on its own, outside a document, where `must_understand` may be `false`."""

    name: ExtensionName
    configuration: NotRequired[Mapping[str, JSONValue]]
    must_understand: NotRequired[bool]


class ZarrV3MandatoryNamedConfigJSON(TypedDict, closed=True):
    """Closed named configuration at an extension point of a document, where understanding is mandatory."""

    name: ExtensionName
    configuration: NotRequired[Mapping[str, JSONValue]]
    must_understand: NotRequired[Literal[True]]


ZarrV3MetadataFieldJSON = ExtensionName | ZarrV3NamedConfigJSON
ZarrV3MandatoryMetadataFieldJSON = ExtensionName | ZarrV3MandatoryNamedConfigJSON
ZarrV3CodecPipelineJSON = Annotated[
    tuple[ZarrV3MandatoryMetadataFieldJSON, ...], Field(min_length=1)
]
ZarrV2FilterPipelineJSON = tuple[ZarrV2CodecMetadata, ...]


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
    storage_transformers: NotRequired[tuple[ZarrV3MandatoryMetadataFieldJSON, ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]


class ZarrV3ConsolidatedMetadataJSON(TypedDict, closed=True):
    """Schema input for the closed inline consolidated-metadata envelope."""

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, ZarrV3ArrayMetadataJSON | ZarrV3GroupMetadataJSON]


class ZarrV3GroupMetadataJSON(TypedDict, extra_items=JSONValue):
    """Schema input for a v3 group document, including arbitrary extensions."""

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, JSONValue]]
    consolidated_metadata: NotRequired[ZarrV3ConsolidatedMetadataJSON]


class ZarrV2ArrayMetadataJSON(TypedDict, extra_items=JSONValue):
    """Schema input for the merged v2 array representation.

    Open, like the runtime validator: the v2 spec says other keys "SHOULD NOT
    be present ... and SHOULD be ignored by implementations"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L91-L92); the group document's "MUST NOT" (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L313) keeps
    `ZarrV2GroupMetadataJSON` closed.
    """

    zarr_format: Literal[2]
    shape: tuple[NonNegativeInt, ...]
    chunks: tuple[NonNegativeInt, ...]
    dtype: ZarrV2DataTypeMetadata
    compressor: ZarrV2CodecMetadata | None
    fill_value: JSONValue
    order: Literal["C", "F"]
    filters: ZarrV2FilterPipelineJSON | None
    dimension_separator: NotRequired[Literal[".", "/"]]
    attributes: NotRequired[Mapping[str, JSONValue]]


class ZarrV2GroupMetadataJSON(TypedDict, closed=True):
    """Schema input for the closed, merged v2 group representation."""

    zarr_format: Literal[2]
    attributes: NotRequired[Mapping[str, JSONValue]]


class ZarrV2ConsolidatedMetadataJSON(TypedDict, closed=True):
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
