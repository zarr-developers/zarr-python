"""Zarr v2 array metadata types."""

from collections.abc import Mapping
from typing import Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.v2.codec import CodecMetadataV2

DataTypeMetadataV2 = str | tuple[tuple[str, str] | tuple[str, str, tuple[int, ...]], ...]
"""The v2 dtype representation.

Either a numpy-style dtype string (e.g. `"<f8"`, `"|S10"`) or a tuple of
field records describing a structured dtype. Each field record is either
a 2-tuple `(name, datatype)` or a 3-tuple `(name, datatype, shape)`
(the 3-tuple form indicates a subarray field).

Endianness is encoded in the prefix character of the dtype string;
parsing it out is a caller concern, not part of this type.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding
"""

ArrayOrderV2 = Literal["C", "F"]
"""Permitted values for the `order` field of v2 array metadata.

`"C"` (row-major) or `"F"` (column-major) — the in-chunk byte layout.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

ArrayDimensionSeparatorV2 = Literal[".", "/"]
"""Permitted values for the `dimension_separator` field of v2 array metadata.

`"."` (legacy default) joins chunk grid coordinates as `0.0`, `0.1`, ...
`"/"` joins them as `0/0`, `0/1`, ... yielding nested directories.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""


class ArrayMetadataV2(TypedDict):
    """
    Zarr v2 array metadata document.

    Models the union of `.zarray` (the spec-defined fields) and `.zattrs`
    (user attributes). On disk, attributes live in a sibling `.zattrs` file
    and are not part of `.zarray`; this type folds them in as the
    `attributes` field so a single TypedDict represents the complete
    in-memory state of a v2 array node. Consumers that read or write a
    real `.zarray` file should split / merge `attributes` accordingly.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DataTypeMetadataV2
    compressor: CodecMetadataV2 | None
    fill_value: object
    order: ArrayOrderV2
    filters: tuple[CodecMetadataV2, ...] | None
    dimension_separator: NotRequired[ArrayDimensionSeparatorV2]
    attributes: Mapping[str, object]
    """User attributes from the sibling `.zattrs` file (not part of `.zarray`).

    See the class docstring for the rationale behind the merged representation.
    """


__all__ = [
    "ArrayDimensionSeparatorV2",
    "ArrayMetadataV2",
    "ArrayOrderV2",
    "DataTypeMetadataV2",
]
