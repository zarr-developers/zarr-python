"""Zarr v2 array metadata types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata import JSON
    from zarr_metadata.v2.codec import NumcodecsConfig


class DataTypeV2Structured(TypedDict):
    """
    A single field entry inside a structured v2 dtype.

    Spec-faithful: ``datatype`` is a numpy-style dtype string; ``shape`` is
    present only when the field is a subarray field.
    """

    fieldname: str
    datatype: str
    shape: NotRequired[tuple[int, ...]]


DataTypeV2 = str | list[DataTypeV2Structured]
"""The v2 dtype representation.

Simple dtypes are numpy-style strings (e.g. ``"<f8"``, ``"|S10"``).
Structured dtypes are lists of field records. Endianness is encoded in the
prefix character of the string; parsing it out is a caller concern, not
part of this type.
"""


class ArrayMetadataV2(TypedDict):
    """
    Zarr v2 array metadata document (the ``.zarray`` content).

    Spec-faithful shape. Attributes live in a sibling ``.zattrs`` file and
    are merged in by convention - the ``attributes`` key here is only used
    in contexts where v2 metadata is represented inline (e.g. consolidated
    metadata).
    """

    zarr_format: Literal[2]
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DataTypeV2
    compressor: NotRequired[NumcodecsConfig | None]
    fill_value: NotRequired[JSON]
    order: NotRequired[Literal["C", "F"]]
    filters: NotRequired[tuple[NumcodecsConfig, ...] | None]
    dimension_separator: NotRequired[Literal[".", "/"]]
    attributes: NotRequired[Mapping[str, JSON]]


__all__ = [
    "ArrayMetadataV2",
    "DataTypeV2",
    "DataTypeV2Structured",
]
