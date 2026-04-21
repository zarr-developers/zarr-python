"""Zarr v3 array metadata types."""

from collections.abc import Mapping
from typing import Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata import JSON, NamedConfig, NamedRequiredConfig


class AllowedExtraField(TypedDict, extra_items=JSON):  # type: ignore[call-arg]
    """
    Extra field on a v3 array metadata document.

    Extras must include ``must_understand: false`` and may carry arbitrary
    additional JSON data.
    """

    must_understand: Literal[False]


# JSON type for a single dimension's rectilinear spec:
# bare int (uniform shorthand), or list of ints / [value, count] RLE pairs.
RectilinearDimSpec = int | list[int | list[int]]


class RegularChunkGridConfig(TypedDict):
    """
    Configuration body of a regular chunk grid.
    """

    chunk_shape: tuple[int, ...]


class RectilinearChunkGridConfig(TypedDict):
    """
    Configuration body of a rectilinear chunk grid.
    """

    kind: Literal["inline"]
    chunk_shapes: tuple[RectilinearDimSpec, ...]


RegularChunkGrid = NamedRequiredConfig[Literal["regular"], RegularChunkGridConfig]
"""Regular chunk grid named-config envelope."""

RectilinearChunkGrid = NamedRequiredConfig[Literal["rectilinear"], RectilinearChunkGridConfig]
"""Rectilinear chunk grid named-config envelope."""


class ArrayMetadataV3(TypedDict, extra_items=AllowedExtraField):  # type: ignore[call-arg]
    """
    Zarr v3 array metadata document (the ``zarr.json`` content for an array).

    Extra keys are permitted if they conform to ``AllowedExtraField``.
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: str | NamedConfig[str, Mapping[str, JSON]]
    shape: tuple[int, ...]
    chunk_grid: str | NamedConfig[str, Mapping[str, JSON]]
    chunk_key_encoding: str | NamedConfig[str, Mapping[str, JSON]]
    fill_value: JSON
    codecs: tuple[str | NamedConfig[str, Mapping[str, JSON]], ...]
    attributes: NotRequired[Mapping[str, JSON]]
    storage_transformers: NotRequired[tuple[str | NamedConfig[str, Mapping[str, JSON]], ...]]
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
