from __future__ import annotations

import functools
import math
import operator
import warnings
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    NotRequired,
    TypedDict,
    cast,
    overload,
)

import numpy as np
from typing_extensions import ReadOnly

from zarr.core.config import config as zarr_config
from zarr.errors import ZarrRuntimeWarning

if TYPE_CHECKING:
    from collections.abc import Iterator


ZARR_JSON = "zarr.json"
ZARRAY_JSON = ".zarray"
ZGROUP_JSON = ".zgroup"
ZATTRS_JSON = ".zattrs"
ZMETADATA_V2_JSON = ".zmetadata"

BytesLike = bytes | bytearray | memoryview
ShapeLike = Iterable[int | np.integer[Any]] | int | np.integer[Any]
ChunksLike = ShapeLike | Sequence[Sequence[int]] | None
# For backwards compatibility
ChunkCoords = tuple[int, ...]
ZarrFormat = Literal[2, 3]
NodeType = Literal["array", "group"]
JSON = str | int | float | bool | Mapping[str, "JSON"] | Sequence["JSON"] | None
MemoryOrder = Literal["C", "F"]
AccessModeLiteral = Literal["r", "r+", "a", "w", "w-"]
ANY_ACCESS_MODE: Final = "r", "r+", "a", "w", "w-"
DimensionNamesLike = Iterable[str | None] | None
DimensionNames = DimensionNamesLike  # for backwards compatibility


class NamedConfig[TName: str, TConfig: Mapping[str, object]](TypedDict):
    """
    A typed dictionary representing an object with a name and configuration, where the configuration
    is an optional mapping of string keys to values, e.g. another typed dictionary or a JSON object.

    This class is generic with two type parameters: the type of the name (``TName``) and the type of
    the configuration (``TConfig``).
    """

    name: ReadOnly[TName]
    """The name of the object."""

    configuration: NotRequired[ReadOnly[TConfig]]
    """The configuration of the object. Not required."""


class NamedRequiredConfig[TName: str, TConfig: Mapping[str, object]](TypedDict):
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


def product(tup: tuple[int, ...]) -> int:
    return functools.reduce(operator.mul, tup, 1)


def ceildiv(a: float, b: float) -> int:
    if a == 0:
        return 0
    return math.ceil(a / b)


def enum_names[E: Enum](enum: type[E]) -> Iterator[str]:
    for item in enum:
        yield item.name


def parse_enum[E: Enum](data: object, cls: type[E]) -> E:
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
    data: JSON | NamedConfig[str, Any], expected_name: str | None = None
) -> tuple[str, dict[str, JSON]]: ...


@overload
def parse_named_configuration(
    data: JSON | NamedConfig[str, Any],
    expected_name: str | None = None,
    *,
    require_configuration: bool = True,
) -> tuple[str, dict[str, JSON] | None]: ...


def parse_named_configuration(
    data: JSON | NamedConfig[str, Any],
    expected_name: str | None = None,
    *,
    require_configuration: bool = True,
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
    """
    Parse a shape-like input into an explicit shape.
    """
    if isinstance(data, int | np.integer):
        if data < 0:
            raise ValueError(f"Expected a non-negative integer. Got {data} instead")
        return (int(data),)
    try:
        data_tuple = tuple(data)
    except TypeError as e:
        msg = f"Expected an integer or an iterable of integers. Got {data} instead."
        raise TypeError(msg) from e

    if not all(isinstance(v, int | np.integer) for v in data_tuple):
        msg = f"Expected an iterable of integers. Got {data} instead."
        raise TypeError(msg)
    if not all(v > -1 for v in data_tuple):
        msg = f"Expected all values to be non-negative. Got {data} instead."
        raise ValueError(msg)

    # cast NumPy scalars to plain python ints
    return tuple(int(x) for x in data_tuple)


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
    """Return the default zarr_format."""
    return cast("ZarrFormat", int(zarr_config.get("default_zarr_format", 3)))


def expand_rle(data: Sequence[int | list[int]]) -> list[int]:
    """Expand a mixed array of bare integers and RLE pairs.

    Per the rectilinear chunk grid spec, each element can be:
    - a bare integer (an explicit edge length)
    - a two-element array ``[value, count]`` (run-length encoded)
    """
    result: list[int] = []
    for item in data:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            val = int(item)
            if val < 1:
                raise ValueError(f"Chunk edge length must be >= 1, got {val}")
            result.append(val)
        elif isinstance(item, list) and len(item) == 2:
            size, count = int(item[0]), int(item[1])
            if size < 1:
                raise ValueError(f"Chunk edge length must be >= 1, got {size}")
            if count < 1:
                raise ValueError(f"RLE repeat count must be >= 1, got {count}")
            result.extend([size] * count)
        else:
            raise ValueError(f"RLE entries must be an integer or [size, count], got {item}")
    return result


def compress_rle(sizes: Sequence[int]) -> list[int | list[int]]:
    """Compress chunk sizes to mixed RLE format per the rectilinear spec.

    Runs of length > 1 are emitted as ``[value, count]`` pairs; runs of
    length 1 are emitted as bare integers::

        [10, 10, 10, 5] -> [[10, 3], 5]
    """
    if not sizes:
        return []
    result: list[int | list[int]] = []
    current = sizes[0]
    count = 1
    for s in sizes[1:]:
        if s == current:
            count += 1
        else:
            result.append([current, count] if count > 1 else current)
            current = s
            count = 1
    result.append([current, count] if count > 1 else current)
    return result


def validate_rectilinear_kind(kind: str | None) -> None:
    """Validate the ``kind`` field of a rectilinear chunk grid configuration.

    The rectilinear spec requires ``kind: "inline"``.
    """
    if kind is None:
        raise ValueError(
            "Rectilinear chunk grid configuration requires a 'kind' field. "
            "Only 'inline' is currently supported."
        )
    if kind != "inline":
        raise ValueError(
            f"Unsupported rectilinear chunk grid kind: {kind!r}. "
            "Only 'inline' is currently supported."
        )


def validate_rectilinear_edges(
    chunk_shapes: Sequence[int | Sequence[int]], array_shape: Sequence[int]
) -> None:
    """Validate that rectilinear chunk edges cover the array extent per dimension.

    Bare-int dimensions (regular step) always cover any extent, so they are
    skipped. Explicit edge lists must sum to at least the array extent.
    """
    for i, (dim_spec, extent) in enumerate(zip(chunk_shapes, array_shape, strict=True)):
        if isinstance(dim_spec, int):
            continue
        edge_sum = sum(dim_spec)
        if edge_sum < extent:
            raise ValueError(
                f"Rectilinear chunk edges for dimension {i} sum to {edge_sum} "
                f"but array shape extent is {extent} (edge sum must be >= extent)"
            )
