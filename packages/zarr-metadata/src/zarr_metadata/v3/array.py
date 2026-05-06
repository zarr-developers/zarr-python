"""Zarr v3 array metadata types."""

from collections.abc import Mapping
from typing import Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.v3._common import MetadataFieldV3


class ExtensionFieldV3(TypedDict, extra_items=object):  # type: ignore[call-arg]
    """
    Required shape of any extension field on a v3 metadata document.

    The Zarr v3 spec permits extra keys on array and group metadata
    documents, provided each value is an object with a `must_understand`
    boolean key. This TypedDict captures that constraint and is used as
    the `extra_items=` parameter on `ArrayMetadataV3` and `GroupMetadataV3`.

    `must_understand` is typed as `bool` rather than `Literal[False]` so
    that applications which understand a particular extension can produce
    or consume it with `must_understand: true` (signalling that readers
    that don't recognize the extension MUST refuse to open the document).
    The common case is still `false`, signalling that unknown readers may
    safely ignore the field.

    Spec interpretation: this type follows the original Zarr v3.0 reading
    of the spec, under which any object with a `must_understand` key is a
    valid extension field. The v3.1 spec rewrite added language requiring
    extension fields to also include a `name: str` key (the "Extension
    definition" form). Under the strict v3.1 reading, real-world extension
    fields written by zarr-python and zarrs (notably `consolidated_metadata`,
    which has no `name` field) are out of spec. The community consensus at
    the time of writing is that this is a regression to be reverted; this
    package models the v3.0 / pre-revert interpretation. See
    https://github.com/zarr-developers/zarr-specs/issues/371 for the
    ongoing discussion.
    """

    must_understand: bool


class ArrayMetadataV3(TypedDict, extra_items=ExtensionFieldV3):  # type: ignore[call-arg]
    """
    Zarr v3 array metadata document (the `zarr.json` content for an array).

    Extra keys are permitted if they conform to `ExtensionFieldV3`.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: MetadataFieldV3
    shape: tuple[int, ...]
    chunk_grid: MetadataFieldV3
    chunk_key_encoding: MetadataFieldV3
    fill_value: object
    codecs: tuple[MetadataFieldV3, ...]
    attributes: NotRequired[Mapping[str, object]]
    storage_transformers: NotRequired[tuple[MetadataFieldV3, ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]


__all__ = [
    "ArrayMetadataV3",
    "ExtensionFieldV3",
]
