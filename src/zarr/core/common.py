from __future__ import annotations

import asyncio
import functools
import math
import operator
import warnings
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from itertools import starmap
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
    NotRequired,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import ReadOnly

from zarr.core.config import config as zarr_config
from zarr.errors import ZarrRuntimeWarning

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator


ZARR_JSON = "zarr.json"
ZARRAY_JSON = ".zarray"
ZGROUP_JSON = ".zgroup"
ZATTRS_JSON = ".zattrs"
ZMETADATA_V2_JSON = ".zmetadata"

BytesLike = bytes | bytearray | memoryview
ShapeLike = Iterable[int] | int
# For backwards compatibility
ChunkCoords: tuple[int, ...]
ZarrFormat = Literal[2, 3]
NodeType = Literal["array", "group"]
JSON = str | int | float | Mapping[str, "JSON"] | Sequence["JSON"] | None
MemoryOrder = Literal["C", "F"]
AccessModeLiteral = Literal["r", "r+", "a", "w", "w-"]
ANY_ACCESS_MODE: Final = "r", "r+", "a", "w", "w-"
DimensionNames = Iterable[str | None] | None

TName = TypeVar("TName", bound=str, covariant=True)
TConfig = TypeVar("TConfig", bound=Mapping[str, object])


class NamedRequiredConfig(TypedDict, Generic[TName, TConfig]):
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


class NamedConfig(TypedDict, Generic[TName, TConfig]):
    """
    A typed dictionary representing an object with a name and configuration, where the configuration
    is a mapping of string keys to values, e.g. another typed dictionary or a JSON object.

    The configuration key is not required.

    This class is generic with two type parameters: the type of the name (``TName``) and the type of
    the configuration (``TConfig``).
    """

    name: ReadOnly[TName]
    """The name of the object."""

    configuration: ReadOnly[NotRequired[TConfig]]
    """The configuration of the object."""


class ArrayMetadataJSON_V2(TypedDict):
    """
    A typed dictionary model for Zarr V2 array metadata.
    """

    zarr_format: Literal[2]
    dtype: str | StructuredName_V2
    shape: Sequence[int]
    chunks: Sequence[int]
    dimension_separator: NotRequired[Literal[".", "/"]]
    fill_value: Any
    filters: Sequence[CodecJSON_V2[str]] | None
    order: Literal["C", "F"]
    compressor: CodecJSON_V2[str] | None
    attributes: NotRequired[Mapping[str, JSON]]


class GroupMetadataJSON_V2(TypedDict):
    """
    A typed dictionary model for Zarr V2 group metadata.
    """

    zarr_format: Literal[2]
    attributes: NotRequired[Mapping[str, JSON]]
    consolidated_metadata: NotRequired[ConsolidatedMetadata_JSON_V2]


class ArrayMetadataJSON_V3(TypedDict):
    """
    A typed dictionary model for Zarr V3 array metadata.
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: str | NamedConfig[str, Mapping[str, object]]
    shape: Sequence[int]
    chunk_grid: NamedConfig[str, Mapping[str, object]]
    chunk_key_encoding: NamedConfig[str, Mapping[str, object]]
    fill_value: object
    codecs: Sequence[str | NamedConfig[str, Mapping[str, object]]]
    attributes: NotRequired[Mapping[str, object]]
    storage_transformers: NotRequired[Sequence[NamedConfig[str, Mapping[str, object]]]]
    dimension_names: NotRequired[Sequence[str | None]]


class GroupMetadataJSON_V3(TypedDict):
    """
    A typed dictionary model for Zarr V3 group metadata.
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, JSON]]
    consolidated_metadata: NotRequired[ConsolidatedMetadata_JSON_V3 | None]


# TODO: use just 1 generic class and parametrize the type of the value type of the metadata
# I.e., ConsolidatedMetadata_JSON[ArrayMetadataJSON_V2 | GroupMetadataJSON_V2]
class ConsolidatedMetadata_JSON_V2(TypedDict):
    """
    A typed dictionary model for Zarr consolidated metadata.

    This model is parameterized by the type of the metadata itself.
    """

    kind: Literal["inline"]
    must_understand: Literal["false"]
    metadata: Mapping[str, ArrayMetadataJSON_V2 | GroupMetadataJSON_V2]


class ConsolidatedMetadata_JSON_V3(TypedDict):
    """
    A typed dictionary model for Zarr consolidated metadata.

    This model is parameterized by the type of the metadata itself.
    """

    kind: Literal["inline"]
    must_understand: Literal["false"]
    metadata: Mapping[str, ArrayMetadataJSON_V3 | GroupMetadataJSON_V3]


class CodecJSON_V2(TypedDict, Generic[TName]):
    """The JSON representation of a codec for Zarr V2"""

    id: ReadOnly[TName]


CodecJSON_V3 = str | NamedConfig[str, Mapping[str, object]]
"""The JSON representation of a codec for Zarr V3."""

# The widest type we will *accept* for a codec JSON
# This covers v2 and v3
CodecJSON = str | Mapping[str, object]
"""The widest type of JSON-like input that could specify a codec."""


# By comparison, The JSON representation of a dtype in zarr v3 is much simpler.
# It's either a string, or a structured dict
DTypeSpec_V3 = str | NamedConfig[str, Mapping[str, object]]

#  This is the JSON representation of a structured dtype in zarr v2
StructuredName_V2 = Sequence["str | StructuredName_V2"]

# This models the type of the name a dtype might have in zarr v2 array metadata
DTypeName_V2 = StructuredName_V2 | str


def product(tup: tuple[int, ...]) -> int:
    return functools.reduce(operator.mul, tup, 1)


def ceildiv(a: float, b: float) -> int:
    if a == 0:
        return 0
    return math.ceil(a / b)


T = TypeVar("T", bound=tuple[Any, ...])
V = TypeVar("V")


async def concurrent_map(
    items: Iterable[T],
    func: Callable[..., Awaitable[V]],
    limit: int | None = None,
) -> list[V]:
    if limit is None:
        return await asyncio.gather(*list(starmap(func, items)))

    else:
        sem = asyncio.Semaphore(limit)

        async def run(item: tuple[Any]) -> V:
            async with sem:
                return await func(*item)

        return await asyncio.gather(*[asyncio.ensure_future(run(item)) for item in items])


E = TypeVar("E", bound=Enum)


def enum_names(enum: type[E]) -> Iterator[str]:
    for item in enum:
        yield item.name


def parse_enum(data: object, cls: type[E]) -> E:
    if isinstance(data, cls):
        return data
    if not isinstance(data, str):
        raise TypeError(f"Expected str, got {type(data)}")
    if data in enum_names(cls):
        return cls(data)
    raise ValueError(f"Value must be one of {list(enum_names(cls))!r}. Got {data} instead.")


def parse_name(data: JSON, expected: str | None = None) -> str:
    if isinstance(data, str):
        if expected is None or data == expected:
            return data
        raise ValueError(f"Expected '{expected}'. Got {data} instead.")
    else:
        raise TypeError(f"Expected a string, got an instance of {type(data)}.")


def parse_configuration(data: JSON) -> JSON:
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    return data


@overload
def parse_named_configuration(
    data: JSON, expected_name: str | None = None
) -> tuple[str, dict[str, JSON]]: ...


@overload
def parse_named_configuration(
    data: JSON, expected_name: str | None = None, *, require_configuration: bool = True
) -> tuple[str, dict[str, JSON] | None]: ...


def parse_named_configuration(
    data: JSON, expected_name: str | None = None, *, require_configuration: bool = True
) -> tuple[str, JSON | None]:
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")
    if "name" not in data:
        raise ValueError(f"Named configuration does not have a 'name' key. Got {data}.")
    name_parsed = parse_name(data["name"], expected_name)
    if "configuration" in data:
        configuration_parsed = parse_configuration(data["configuration"])
    elif require_configuration:
        raise ValueError(f"Named configuration does not have a 'configuration' key. Got {data}.")
    else:
        configuration_parsed = None
    return name_parsed, configuration_parsed


def parse_shapelike(data: ShapeLike) -> tuple[int, ...]:
    if isinstance(data, int):
        if data < 0:
            raise ValueError(f"Expected a non-negative integer. Got {data} instead")
        return (data,)
    try:
        data_tuple = tuple(data)
    except TypeError as e:
        msg = f"Expected an integer or an iterable of integers. Got {data} instead."
        raise TypeError(msg) from e

    if not all(isinstance(v, int) for v in data_tuple):
        msg = f"Expected an iterable of integers. Got {data} instead."
        raise TypeError(msg)
    if not all(v > -1 for v in data_tuple):
        msg = f"Expected all values to be non-negative. Got {data} instead."
        raise ValueError(msg)
    return data_tuple


def parse_fill_value(data: Any) -> Any:
    # todo: real validation
    return data


def parse_order(data: Any) -> Literal["C", "F"]:
    if data in ("C", "F"):
        return cast("Literal['C', 'F']", data)
    raise ValueError(f"Expected one of ('C', 'F'), got {data} instead.")


def parse_bool(data: Any) -> bool:
    if isinstance(data, bool):
        return data
    raise ValueError(f"Expected bool, got {data} instead.")


def _warn_write_empty_chunks_kwarg() -> None:
    # TODO: link to docs page on array configuration in this message
    msg = (
        "The `write_empty_chunks` keyword argument is deprecated and will be removed in future versions. "
        "To control whether empty chunks are written to storage, either use the `config` keyword "
        "argument, as in `config={'write_empty_chunks': True}`,"
        "or change the global 'array.write_empty_chunks' configuration variable."
    )
    warnings.warn(msg, ZarrRuntimeWarning, stacklevel=2)


def _warn_order_kwarg() -> None:
    # TODO: link to docs page on array configuration in this message
    msg = (
        "The `order` keyword argument has no effect for Zarr format 3 arrays. "
        "To control the memory layout of the array, either use the `config` keyword "
        "argument, as in `config={'order': 'C'}`,"
        "or change the global 'array.order' configuration variable."
    )
    warnings.warn(msg, ZarrRuntimeWarning, stacklevel=2)


def _default_zarr_format() -> ZarrFormat:
    """Return the default zarr_version"""
    return cast("ZarrFormat", int(zarr_config.get("default_zarr_format", 3)))
