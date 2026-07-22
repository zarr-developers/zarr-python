"""Zarr v3 array metadata types."""

from collections.abc import Mapping
from typing import Literal, NotRequired, TypeAlias

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON

ZarrV3ExtensionField: TypeAlias = JSONValue
"""The JSON value of an unknown top-level v3 metadata field.

An object carrying the literal member `must_understand: false` may be ignored.
Every other JSON shape implicitly requires understanding; recognition itself
belongs to the reader rather than this structural type.
"""


class ZarrV3ArrayMetadataJSON(TypedDict, extra_items=ZarrV3ExtensionField):
    """
    Zarr v3 array metadata document (the `zarr.json` content for an array).

    Extra keys may contain arbitrary JSON values.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: ZarrV3MetadataFieldJSON
    shape: tuple[int, ...]
    chunk_grid: ZarrV3MetadataFieldJSON
    chunk_key_encoding: ZarrV3MetadataFieldJSON
    fill_value: JSONValue
    codecs: tuple[ZarrV3MetadataFieldJSON, ...]
    attributes: NotRequired[Mapping[str, JSONValue]]
    storage_transformers: NotRequired[tuple[ZarrV3MetadataFieldJSON, ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]


class ZarrV3ArrayMetadataJSONPartial(TypedDict, total=False, extra_items=ZarrV3ExtensionField):
    """
    Partial form of `ZarrV3ArrayMetadataJSON`: every field is `NotRequired`.

    Field annotations and `extra_items=` mirror `ZarrV3ArrayMetadataJSON` exactly.
    The only difference is `total=False`, which makes every key optional
    at the type level.

    Use this when typing dicts that intentionally hold a subset of a complete
    v3 array metadata document — e.g. test fixtures that override only a few
    fields of a base template, or callers that build a fragment to be merged
    into a complete document elsewhere.

    The `NotRequired[...]` wrappers on `attributes`, `storage_transformers`,
    and `dimension_names` are intentional: keeping them preserves byte-identical
    `__annotations__` with `ZarrV3ArrayMetadataJSON` so the `==` check in
    `tests/test_partial_equivalence.py` passes without special-casing those
    fields (PEP 655 explicitly permits `NotRequired` inside `total=False`).

    Drift between this type and `ZarrV3ArrayMetadataJSON` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: ZarrV3MetadataFieldJSON
    shape: tuple[int, ...]
    chunk_grid: ZarrV3MetadataFieldJSON
    chunk_key_encoding: ZarrV3MetadataFieldJSON
    fill_value: JSONValue
    codecs: tuple[ZarrV3MetadataFieldJSON, ...]
    attributes: NotRequired[Mapping[str, JSONValue]]
    storage_transformers: NotRequired[tuple[ZarrV3MetadataFieldJSON, ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]


__all__ = [
    "ZarrV3ArrayMetadataJSON",
    "ZarrV3ArrayMetadataJSONPartial",
    "ZarrV3ExtensionField",
]
