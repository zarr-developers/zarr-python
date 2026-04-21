"""Zarr v2 array metadata types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.common import JSON
    from zarr_metadata.v2.codec import NumcodecsConfig


class DataTypeV2Structured(TypedDict):
    """
    A single field entry inside a structured v2 dtype.

    Spec-faithful: `datatype` is a numpy-style dtype string; `shape` is
    present only when the field is a subarray field.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding
    """

    fieldname: str
    datatype: str
    shape: NotRequired[tuple[int, ...]]


DataTypeV2 = str | tuple[DataTypeV2Structured, ...]
"""The v2 dtype representation.

Simple dtypes are numpy-style strings (e.g. ``"<f8"``, ``"|S10"``).
Structured dtypes are lists of field records. Endianness is encoded in the
prefix character of the string; parsing it out is a caller concern, not
part of this type.
"""

class ArrayMetadataV2(TypedDict):
    """
    Zarr v2 array metadata document (the `.zarray` content).

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DataTypeV2
    compressor: NumcodecsConfig | None
    fill_value: JSON
    order: Literal["C", "F"]
    filters: tuple[NumcodecsConfig, ...] | None
    dimension_separator: NotRequired[Literal[".", "/"]]



__all__ = [
    "ArrayMetadataV2",
    "DataTypeV2",
    "DataTypeV2Structured",
]
