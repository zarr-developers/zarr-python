"""Zarr v3 array metadata types."""

from collections.abc import Mapping
from typing import Literal, NotRequired, TypeAlias

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v3._common import MetadataV3

ExtensionFieldV3: TypeAlias = JSONValue
"""The JSON value of an unknown top-level v3 metadata field.

An object carrying the literal member `must_understand: false` may be ignored.
Every other JSON shape implicitly requires understanding; recognition itself
belongs to the reader rather than this structural type.
"""


class ArrayMetadataV3(TypedDict, extra_items=ExtensionFieldV3):
    """
    Zarr v3 array metadata document (the `zarr.json` content for an array).

    Extra keys may contain arbitrary JSON values.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: MetadataV3
    shape: tuple[int, ...]
    chunk_grid: MetadataV3
    chunk_key_encoding: MetadataV3
    fill_value: JSONValue
    codecs: tuple[MetadataV3, ...]
    attributes: NotRequired[Mapping[str, JSONValue]]
    storage_transformers: NotRequired[tuple[MetadataV3, ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]


class ArrayMetadataV3Partial(TypedDict, total=False, extra_items=ExtensionFieldV3):
    """
    Partial form of `ArrayMetadataV3`: every field is `NotRequired`.

    Field annotations and `extra_items=` mirror `ArrayMetadataV3` exactly.
    The only difference is `total=False`, which makes every key optional
    at the type level.

    Use this when typing dicts that intentionally hold a subset of a complete
    v3 array metadata document — e.g. test fixtures that override only a few
    fields of a base template, or callers that build a fragment to be merged
    into a complete document elsewhere.

    The `NotRequired[...]` wrappers on `attributes`, `storage_transformers`,
    and `dimension_names` are intentional: keeping them preserves byte-identical
    `__annotations__` with `ArrayMetadataV3` so the `==` check in
    `tests/test_partial_equivalence.py` passes without special-casing those
    fields (PEP 655 explicitly permits `NotRequired` inside `total=False`).

    Drift between this type and `ArrayMetadataV3` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: MetadataV3
    shape: tuple[int, ...]
    chunk_grid: MetadataV3
    chunk_key_encoding: MetadataV3
    fill_value: JSONValue
    codecs: tuple[MetadataV3, ...]
    attributes: NotRequired[Mapping[str, JSONValue]]
    storage_transformers: NotRequired[tuple[MetadataV3, ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]


__all__ = [
    "ArrayMetadataV3",
    "ArrayMetadataV3Partial",
    "ExtensionFieldV3",
]
