"""Zarr v3 array metadata types."""

from collections.abc import Mapping
from typing import Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.v3._common import MetadataField


class ExtraField(TypedDict, extra_items=object):  # type: ignore[call-arg]
    """
    Extra field on a v3 array metadata document.

    Extras must include `must_understand: false` and may carry arbitrary
    additional JSON data.
    """

    must_understand: Literal[False]


class ArrayMetadataV3(TypedDict, extra_items=ExtraField):  # type: ignore[call-arg]
    """
    Zarr v3 array metadata document (the `zarr.json` content for an array).

    Extra keys are permitted if they conform to `ExtraField`.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: MetadataField
    shape: tuple[int, ...]
    chunk_grid: MetadataField
    chunk_key_encoding: MetadataField
    fill_value: object
    codecs: tuple[MetadataField, ...]
    attributes: NotRequired[Mapping[str, object]]
    storage_transformers: NotRequired[tuple[MetadataField, ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]


__all__ = [
    "ArrayMetadataV3",
    "ExtraField",
]
