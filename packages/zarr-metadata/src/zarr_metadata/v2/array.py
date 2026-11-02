"""Zarr v2 array metadata types."""

from collections.abc import Mapping
from typing import Annotated, Final, Literal, NotRequired

from annotated_types import Ge
from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v2._field_json import ZarrV2CodecMetadata, ZarrV2DataTypeMetadata

ZarrV2ArrayOrder = Literal["C", "F"]
"""Literal type of permitted values for the `order` field of v2 array metadata.

`"C"` (row-major) or `"F"` (column-major) — the in-chunk byte layout.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

ZARR_V2_ARRAY_ORDER: Final = ("C", "F")
"""Tuple of permitted values for the `order` field of v2 array metadata."""

ZarrV2ArrayDimensionSeparator = Literal[".", "/"]
"""Literal type of permitted values for the `dimension_separator` field of v2 array metadata.

`"."` (legacy default) joins chunk grid coordinates as `0.0`, `0.1`, ...
`"/"` joins them as `0/0`, `0/1`, ... yielding nested directories.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

ZARR_V2_ARRAY_DIMENSION_SEPARATOR: Final = (".", "/")
"""Tuple of permitted values for the `dimension_separator` field of v2 array metadata."""


class ZarrV2ZArrayJSON(TypedDict):
    """
    On-disk `.zarray` file content.

    Strict shape of the JSON document persisted at `<path>/.zarray` for
    a v2 array. User attributes live in a sibling `.zattrs` file and are
    NOT part of this type; see `ZarrV2ZAttrsJSON`.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]
    shape: tuple[Annotated[int, Ge(0)], ...]
    chunks: tuple[Annotated[int, Ge(0)], ...]
    dtype: ZarrV2DataTypeMetadata
    compressor: ZarrV2CodecMetadata | None
    fill_value: JSONValue
    order: ZarrV2ArrayOrder
    filters: tuple[ZarrV2CodecMetadata, ...] | None
    dimension_separator: NotRequired[ZarrV2ArrayDimensionSeparator]


class ZarrV2ArrayMetadataJSON(TypedDict, extra_items=JSONValue):
    """
    Zarr v2 array metadata document, in-memory merged form.

    Models the union of `.zarray` (the spec-defined fields, https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L51-L92)
    and `.zattrs` (user attributes, https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330). On disk, attributes live in a sibling `.zattrs` file
    and are not part of `.zarray`; this type folds them in as the
    `attributes` field so a single TypedDict represents the complete
    in-memory state of a v2 array node. Consumers that read or write a
    real `.zarray` file should split / merge `attributes` accordingly,
    or use `ZarrV2ZArrayJSON` (strict on-disk) plus `ZarrV2ZAttrsJSON` directly.

    The type is open: other keys "SHOULD NOT be present ... and SHOULD be ignored by
    implementations" (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L91-L92), and each
    other member is a JSON value.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]
    shape: tuple[Annotated[int, Ge(0)], ...]
    chunks: tuple[Annotated[int, Ge(0)], ...]
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


ZarrV2ArrayMetadataStoreKey = Literal[".zarray"]
"""Literal type of the store key holding a v2 array's metadata document."""

ZARR_V2_ARRAY_METADATA_STORE_KEY: Final[ZarrV2ArrayMetadataStoreKey] = ".zarray"
"""The store key a v2 array's metadata document is persisted under."""


__all__ = [
    "ZARR_V2_ARRAY_DIMENSION_SEPARATOR",
    "ZARR_V2_ARRAY_METADATA_STORE_KEY",
    "ZARR_V2_ARRAY_ORDER",
    "ZarrV2ArrayDimensionSeparator",
    "ZarrV2ArrayMetadataJSON",
    "ZarrV2ArrayMetadataStoreKey",
    "ZarrV2ArrayOrder",
    "ZarrV2ZArrayJSON",
]
