"""Zarr v3 array metadata types."""

from collections.abc import Mapping
from typing import Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.common import JSON, NamedConfig


class AllowedExtraField(TypedDict, extra_items=JSON):  # type: ignore[call-arg]
    """
    Extra field on a v3 array metadata document.

    Extras must include ``must_understand: false`` and may carry arbitrary
    additional JSON data.
    """

    must_understand: Literal[False]


# JSON type for a single dimension's rectilinear spec:
# bare int (uniform shorthand), or list of ints / [value, count] RLE pairs.
RectilinearDimSpec = int | tuple[int | tuple[int, int], ...]

MetadataField = str | NamedConfig[str, Mapping[str, JSON]]
"""A string or a {name: str, configuration: {...}} key value pair"""
class RegularChunkGridConfig(TypedDict):
    """
    Configuration body of a regular chunk grid.

    See #SPEC_URL
    """
    chunk_shape: tuple[int, ...]


class RectilinearChunkGridConfig(TypedDict):
    """
    Configuration body of a rectilinear chunk grid.

    See #SPEC_URL
    """
    kind: Literal["inline"]
    chunk_shapes: tuple[RectilinearDimSpec, ...]


class RegularChunkGrid(TypedDict):
    """Regular chunk grid named-config envelope."""
    name: Literal["regular"]
    configuration: NotRequired[RegularChunkGridConfig]


class RectilinearChunkGrid(TypedDict):
    """Rectilinear chunk grid named-config envelope."""
    name: Literal["rectilinear"]
    configuration: RectilinearChunkGridConfig


class ArrayMetadataV3(TypedDict, extra_items=AllowedExtraField):  # type: ignore[call-arg]
    """
    Zarr v3 array metadata document (the ``zarr.json`` content for an array).

    Extra keys are permitted if they conform to ``AllowedExtraField``.
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: MetadataField
    shape: tuple[int, ...]
    chunk_grid: MetadataField
    chunk_key_encoding: MetadataField
    fill_value: JSON
    codecs: tuple[MetadataField, ...]
    attributes: NotRequired[Mapping[str, JSON]]
    storage_transformers: NotRequired[tuple[MetadataField, ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]


__all__ = [
    "AllowedExtraField",
    "ArrayMetadataV3",
    "RectilinearChunkGrid",
    "RectilinearChunkGridConfig",
    "RectilinearDimSpec",
    "RegularChunkGrid",
    "RegularChunkGridConfig",
]
