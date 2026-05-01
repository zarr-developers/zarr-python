"""Zarr v3 array metadata types."""

from collections.abc import Mapping
from typing import Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.common import JSON, NamedConfig


class ExtraField(TypedDict, extra_items=JSON):  # type: ignore[call-arg]
    """
    Extra field on a v3 array metadata document.

    Extras must include `must_understand: false` and may carry arbitrary
    additional JSON data.
    """

    must_understand: Literal[False]


MetadataField = str | NamedConfig
"""A string or a {name: str, configuration: {...}} key value pair, where the 'configuration' key may be omitted. """


class ArrayMetadataV3(TypedDict, extra_items=ExtraField):  # type: ignore[call-arg]
    """
    Zarr v3 array metadata document (the `zarr.json` content for an array).

    Extra keys are permitted if they conform to `AllowedExtraField`.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata
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
    "ArrayMetadataV3",
    "ExtraField",
    "MetadataField",
]
