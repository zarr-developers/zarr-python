"""Zarr v2 array metadata types."""

from collections.abc import Mapping
from typing import Literal, NotRequired, TypedDict

from zarr_metadata.v2.codec import NumcodecsConfig

DataTypeV2Structured = tuple[str, str] | tuple[str, str, tuple[int, ...]]
"""
A single field entry inside a structured v2 dtype.

Spec-faithful: `datatype` is a numpy-style dtype string; `shape` is
present only when the field is a subarray field.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding
"""

DataTypeV2 = str | tuple[DataTypeV2Structured, ...]
"""The v2 dtype representation.

Simple dtypes are numpy-style strings (e.g. `"<f8"`, `"|S10"`).
Structured dtypes are lists of field records. Endianness is encoded in the
prefix character of the string; parsing it out is a caller concern, not
part of this type.
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
    dtype: DataTypeV2
    compressor: NumcodecsConfig | None
    fill_value: object
    order: Literal["C", "F"]
    filters: tuple[NumcodecsConfig, ...] | None
    dimension_separator: NotRequired[Literal[".", "/"]]
    attributes: Mapping[str, object]
    """User attributes from the sibling `.zattrs` file (not part of `.zarray`).

    See the class docstring for the rationale behind the merged representation.
    """


__all__ = [
    "ArrayMetadataV2",
    "DataTypeV2",
    "DataTypeV2Structured",
]
