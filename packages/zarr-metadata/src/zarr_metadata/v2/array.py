"""Zarr v2 array metadata types."""

from collections.abc import Mapping
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v2.codec import ZarrV2CodecMetadata

ZarrV2DataTypeMetadata = str | tuple[tuple[str, str] | tuple[str, str, tuple[int, ...]], ...]
"""The v2 dtype representation.

Either a numpy-style dtype string (e.g. `"<f8"`, `"|S10"`) or a tuple of
field records describing a structured dtype. Each field record is either
a 2-tuple `(name, datatype)` or a 3-tuple `(name, datatype, shape)`
(the 3-tuple form indicates a subarray field).

Endianness is encoded in the prefix character of the dtype string;
parsing it out is a caller concern, not part of this type.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding
"""

ZarrV2ArrayOrder = Literal["C", "F"]
"""Literal type of permitted values for the `order` field of v2 array metadata.

`"C"` (row-major) or `"F"` (column-major) — the in-chunk byte layout.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

ARRAY_ORDER_V2: Final = ("C", "F")
"""Tuple of permitted values for the `order` field of v2 array metadata."""

ZarrV2ArrayDimensionSeparator = Literal[".", "/"]
"""Literal type of permitted values for the `dimension_separator` field of v2 array metadata.

`"."` (legacy default) joins chunk grid coordinates as `0.0`, `0.1`, ...
`"/"` joins them as `0/0`, `0/1`, ... yielding nested directories.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

ARRAY_DIMENSION_SEPARATOR_V2: Final = (".", "/")
"""Tuple of permitted values for the `dimension_separator` field of v2 array metadata."""


class ZArrayMetadata(TypedDict):
    """
    On-disk `.zarray` file content.

    Strict shape of the JSON document persisted at `<path>/.zarray` for
    a v2 array. User attributes live in a sibling `.zattrs` file and are
    NOT part of this type; see `ZAttrsMetadata`.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: ZarrV2DataTypeMetadata
    compressor: ZarrV2CodecMetadata | None
    fill_value: JSONValue
    order: ZarrV2ArrayOrder
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    dimension_separator: NotRequired[ZarrV2ArrayDimensionSeparator]


class ZarrV2ArrayMetadataJSON(TypedDict):
    """
    Zarr v2 array metadata document, in-memory merged form.

    Models the union of `.zarray` (the spec-defined fields) and `.zattrs`
    (user attributes). On disk, attributes live in a sibling `.zattrs` file
    and are not part of `.zarray`; this type folds them in as the
    `attributes` field so a single TypedDict represents the complete
    in-memory state of a v2 array node. Consumers that read or write a
    real `.zarray` file should split / merge `attributes` accordingly,
    or use `ZArrayMetadata` (strict on-disk) plus `ZAttrsMetadata` directly.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: ZarrV2DataTypeMetadata
    compressor: ZarrV2CodecMetadata | None
    fill_value: JSONValue
    order: ZarrV2ArrayOrder
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    dimension_separator: NotRequired[ZarrV2ArrayDimensionSeparator]
    attributes: NotRequired[Mapping[str, JSONValue]]
    """User attributes from the sibling `.zattrs` file (not part of `.zarray`).

    See the class docstring for the rationale behind the merged representation.
    """


class ZarrV2ArrayMetadataJSONPartial(TypedDict, total=False):
    """
    Partial form of `ZarrV2ArrayMetadataJSON`: every field is `NotRequired`.

    Field annotations mirror `ZarrV2ArrayMetadataJSON` exactly. The only difference is
    `total=False`, which makes every key optional at the type level.

    Use this when typing dicts that intentionally hold a subset of a complete
    v2 array metadata document — e.g. test fixtures that override only a few
    fields of a base template, or callers that build a fragment to be merged
    into a complete document elsewhere.

    The `NotRequired[...]` wrappers on `dimension_separator` and `attributes`
    are intentional: keeping them preserves byte-identical `__annotations__`
    with `ZarrV2ArrayMetadataJSON` so the `==` check in
    `tests/test_partial_equivalence.py` passes without special-casing those
    fields (PEP 655 explicitly permits `NotRequired` inside `total=False`).

    Note: v2 array metadata has no `extra_items` setting (the v2 spec has no
    extension-field concept), so this partial inherits the same closed shape.

    Drift between this type and `ZarrV2ArrayMetadataJSON` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[2]
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: ZarrV2DataTypeMetadata
    compressor: ZarrV2CodecMetadata | None
    fill_value: JSONValue
    order: ZarrV2ArrayOrder
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    dimension_separator: NotRequired[ZarrV2ArrayDimensionSeparator]
    attributes: NotRequired[Mapping[str, JSONValue]]
    """User attributes from the sibling `.zattrs` file (not part of `.zarray`).

    See the class docstring for the rationale behind the merged representation.
    """


__all__ = [
    "ARRAY_DIMENSION_SEPARATOR_V2",
    "ARRAY_ORDER_V2",
    "ZArrayMetadata",
    "ZarrV2ArrayDimensionSeparator",
    "ZarrV2ArrayMetadataJSON",
    "ZarrV2ArrayMetadataJSONPartial",
    "ZarrV2ArrayOrder",
    "ZarrV2DataTypeMetadata",
]
