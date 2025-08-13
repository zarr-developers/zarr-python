"""Type definitions and typed constants for Zarr data structures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Final, Generic, Literal, NotRequired, TypeVar

from typing_extensions import ReadOnly, TypedDict

NodeType = Literal["array", "group"]
"""The names of the nodes in a Zarr hierarchy."""


BytesLike = bytes | bytearray | memoryview

ChunkCoords = tuple[int, ...]

ShapeLike = tuple[int, ...] | int
"""A shape, either as a tuple of integers or a single integer."""

JSON = str | int | float | Mapping[str, "JSON"] | Sequence["JSON"] | None
"""A JSON value."""

MemoryOrder = Literal["C", "F"]
"""Memory order represented as a string using NumPy / Zarr V2 conventions."""

MEMORY_ORDER: Final = "C", "F"

DimensionSeparator_V2 = Literal[".", "/"]
"""The possible values for the dimension separator in Zarr V2 array metadata."""

ChunkCoordsLike = Iterable[int]
AccessModeLiteral = Literal["r", "r+", "a", "w", "w-"]
ACCESS_MODE_LITERAL: Final = "r", "r+", "a", "w", "w-"

DimensionNames = Iterable[str | None] | None
"""The possible types for the dimension names in Zarr V3 array metadata."""


TName = TypeVar("TName", bound=str)
TConfig = TypeVar("TConfig", bound=Mapping[str, object])


class NamedConfig(TypedDict, Generic[TName, TConfig]):
    """
    A typed dictionary representing an object with a name and configuration, where the configuration
    is a mapping of string keys to values, e.g. another typed dictionary or a JSON object.

    This class is generic with two type parameters: the type of the name (``TName``) and the type of
    the configuration (``TConfig``).
    """

    name: ReadOnly[TName]
    """The name of the object."""

    configuration: ReadOnly[TConfig]
    """The configuration of the object."""


ZARR_JSON: Final = "zarr.json"
ZARRAY_JSON: Final = ".zarray"
ZGROUP_JSON: Final = ".zgroup"
ZATTRS_JSON: Final = ".zattrs"
ZMETADATA_V2_JSON: Final = ".zmetadata"


class CodecJSON_V2(TypedDict, Generic[TName]):
    """The JSON representation of a codec for Zarr V2"""

    id: ReadOnly[TName]


class GroupMetadataJSON_V2(TypedDict):
    """
    A typed dictionary model for Zarr format 2 group metadata.
    """

    zarr_format: Literal[2]


class ArrayMetadataJSON_V2(TypedDict):
    """
    A typed dictionary model for Zarr format 2 metadata.
    """

    zarr_format: Literal[2]
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: str | tuple[tuple[str, object], ...]
    compressor: CodecJSON_V2[str] | None
    fill_value: object | None
    order: MemoryOrder
    filters: tuple[CodecJSON_V2[str], ...] | None
    dimension_separator: NotRequired[DimensionSeparator_V2]


class GroupMetadataJSON_V3(TypedDict):
    """
    A typed dictionary model for Zarr format 2 group metadata.
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, JSON]]


class ArrayMetadataJSON_V3(TypedDict):
    """
    A typed dictionary model for zarr v3 metadata.
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: str | NamedConfig[str, Mapping[str, object]]
    shape: tuple[int, ...]
    chunk_grid: NamedConfig[str, Mapping[str, object]]
    chunk_key_encoding: NamedConfig[str, Mapping[str, object]]
    fill_value: object
    codecs: tuple[str | NamedConfig[str, Mapping[str, object]], ...]
    attributes: NotRequired[Mapping[str, JSON]]
    storage_transformers: NotRequired[tuple[NamedConfig[str, Mapping[str, object]], ...]]
    dimension_names: NotRequired[tuple[str | None]]
