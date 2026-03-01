from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr.core.chunk_grids.common import (
    ChunkEdgeLength,
    ChunkGrid,
    ChunksLike,
    ResolvedChunkSpec,
    _auto_partition,
    _guess_chunks,
    _guess_num_chunks_per_axis_shard,
    _is_nested_sequence,
    _normalize_chunks,
)
from zarr.core.chunk_grids.rectilinear import (
    RectilinearChunkGrid,
    RectilinearChunkGridConfigurationDict,
    _compress_run_length_encoding,
    _expand_run_length_encoding,
    _normalize_rectilinear_chunks,
    _parse_chunk_shapes,
)
from zarr.core.chunk_grids.regular import RegularChunkGrid
from zarr.core.common import JSON, NamedConfig, parse_named_configuration, parse_shapelike
from zarr.registry import get_chunk_grid_class, register_chunk_grid

if TYPE_CHECKING:
    from zarr.core.array import ShardsLike

register_chunk_grid("regular", RegularChunkGrid)
register_chunk_grid("rectilinear", RectilinearChunkGrid)

__all__ = [
    "ChunkEdgeLength",
    "ChunkGrid",
    "ChunksLike",
    "RectilinearChunkGrid",
    "RectilinearChunkGridConfigurationDict",
    "RegularChunkGrid",
    "ResolvedChunkSpec",
    "_auto_partition",
    "_compress_run_length_encoding",
    "_expand_run_length_encoding",
    "_guess_chunks",
    "_guess_num_chunks_per_axis_shard",
    "_is_nested_sequence",
    "_normalize_chunks",
    "_normalize_rectilinear_chunks",
    "_parse_chunk_shapes",
    "parse_chunk_grid",
    "parse_chunk_grid_from_dict",
    "resolve_chunk_spec",
]


def parse_chunk_grid_from_dict(
    data: dict[str, JSON] | ChunkGrid | NamedConfig[str, Any],
) -> ChunkGrid:
    """
    Parse a chunk grid from a dictionary or return an existing ChunkGrid instance.

    This is the standalone factory function for creating ChunkGrid instances from
    serialized metadata (dictionaries). It uses the chunk grid registry to look up
    the appropriate class by name.

    Parameters
    ----------
    data : dict[str, JSON] | ChunkGrid | NamedConfig[str, Any]
        Either a ChunkGrid instance (returned as-is) or a dictionary with
        'name' and 'configuration' keys.

    Returns
    -------
    ChunkGrid
        A concrete ChunkGrid instance.

    Raises
    ------
    ValueError
        If the chunk grid name is not found in the registry.
    """
    if isinstance(data, ChunkGrid):
        return data

    name_parsed, _ = parse_named_configuration(data)
    try:
        chunk_grid_cls = get_chunk_grid_class(name_parsed)
    except KeyError as e:
        raise ValueError(f"Unknown chunk grid. Got {name_parsed}.") from e
    return chunk_grid_cls.from_dict(data)  # type: ignore[arg-type]


def parse_chunk_grid(
    chunks: ChunksLike,
    *,
    shape: tuple[int, ...] | Any,
    item_size: int = 1,
    zarr_format: int | None = None,
) -> ChunkGrid:
    """
    Parse a chunks parameter into a ChunkGrid instance.

    This function handles multiple input formats for the chunks parameter and always
    returns a concrete ChunkGrid instance:
    - ChunkGrid instances: Returned as-is
    - Nested sequences (e.g., [[10, 20], [5, 5]]): Converted to RectilinearChunkGrid
      (Zarr v3 only)
    - Nested sequences with RLE (e.g., [[[10, 6]], [[10, 5]]]): Expanded and converted to
      RectilinearChunkGrid
    - Regular tuples/ints (e.g., (10, 10) or 10): Converted to RegularChunkGrid
    - Literal "auto": Computed using auto-chunking heuristics and converted to RegularChunkGrid

    Parameters
    ----------
    chunks : ChunksLike
        The chunks parameter to parse. Can be:
        - A ChunkGrid instance
        - A nested sequence for variable-sized chunks (supports RLE format)
        - A tuple of integers for uniform chunks
        - A single integer (for 1D arrays or uniform chunks across all dimensions)
        - The literal "auto"

        RLE (Run-Length Encoding) format: [[value, count]] expands to 'count' repetitions
        of 'value'.
        Example: [[[10, 6]]] creates 6 chunks of size 10 each.
    shape : ShapeLike
        The shape of the array. Required to create RegularChunkGrid for "auto" or tuple inputs.
    item_size : int, default=1
        The size of each array element in bytes. Used for auto-chunking heuristics.
    zarr_format : {2, 3, None}, optional
        The Zarr format version. Required for validating nested sequences
        (which are only supported in Zarr v3).

    Returns
    -------
    ChunkGrid
        A concrete ChunkGrid instance (either RegularChunkGrid or RectilinearChunkGrid).

    Raises
    ------
    ValueError
        If nested sequences are used with zarr_format=2, or if variable chunks don't sum
        to shape.
    TypeError
        If the chunks parameter cannot be parsed.

    Examples
    --------
    >>> # ChunkGrid instance
    >>> from zarr.core.chunk_grids import RegularChunkGrid
    >>> grid = RegularChunkGrid(chunk_shape=(10, 10))
    >>> result = parse_chunk_grid(grid, shape=(100, 100))
    >>> result is grid
    True

    >>> # Nested sequence for RectilinearChunkGrid
    >>> result = parse_chunk_grid([[10, 20, 30], [5, 5]], shape=(60, 10), zarr_format=3)
    >>> type(result).__name__
    'RectilinearChunkGrid'
    >>> result.chunk_shapes
    ((10, 20, 30), (5, 5))

    >>> # RLE format for RectilinearChunkGrid
    >>> result = parse_chunk_grid([[[10, 6]], [[10, 5]]], shape=(60, 50), zarr_format=3)
    >>> type(result).__name__
    'RectilinearChunkGrid'
    >>> result.chunk_shapes
    ((10, 10, 10, 10, 10, 10), (10, 10, 10, 10, 10))

    >>> # Regular tuple
    >>> result = parse_chunk_grid((10, 10), shape=(100, 100))
    >>> type(result).__name__
    'RegularChunkGrid'
    >>> result.chunk_shape
    (10, 10)

    >>> # Literal "auto"
    >>> result = parse_chunk_grid("auto", shape=(100, 100), item_size=4)
    >>> type(result).__name__
    'RegularChunkGrid'
    >>> isinstance(result.chunk_shape, tuple)
    True

    >>> # Single int
    >>> result = parse_chunk_grid(10, shape=(100, 100))
    >>> result.chunk_shape
    (10, 10)
    """

    # Case 1: Already a ChunkGrid instance
    if isinstance(chunks, ChunkGrid):
        return chunks

    # Parse shape to ensure it's a tuple
    shape_parsed = parse_shapelike(shape)

    # Case 2: String "auto" -> RegularChunkGrid
    if isinstance(chunks, str):
        # chunks can only be "auto" based on type annotation
        # _normalize_chunks expects None or True for auto-chunking, not "auto"
        chunk_shape = _normalize_chunks(None, shape_parsed, item_size)
        return RegularChunkGrid(chunk_shape=chunk_shape)

    # Case 3: Single int -> RegularChunkGrid
    if isinstance(chunks, int):
        chunk_shape = _normalize_chunks(chunks, shape_parsed, item_size)
        return RegularChunkGrid(chunk_shape=chunk_shape)

    # Case 4: Tuple or sequence - determine if regular or variable chunks
    if _is_nested_sequence(chunks):
        # Variable chunks (nested sequence) -> RectilinearChunkGrid
        if zarr_format == 2:
            raise ValueError(
                "Variable chunks (nested sequences) are only supported in Zarr format 3. "
                "Use zarr_format=3 or provide a regular tuple for chunks."
            )

        # Normalize and validate variable chunks
        chunk_shapes = _normalize_rectilinear_chunks(chunks, shape_parsed)  # type: ignore[arg-type]
        return RectilinearChunkGrid(chunk_shapes=chunk_shapes)
    else:
        # Regular tuple of ints -> RegularChunkGrid
        chunk_shape = _normalize_chunks(chunks, shape_parsed, item_size)
        return RegularChunkGrid(chunk_shape=chunk_shape)


def _validate_zarr_format_compatibility(
    chunks: Any,
    shards: Any,
    zarr_format: int,
) -> None:
    """
    Validate that chunk specification is compatible with Zarr format.

    Parameters
    ----------
    chunks : Any
        The chunks specification.
    shards : Any
        The shards specification.
    zarr_format : {2, 3}
        The Zarr format version.

    Raises
    ------
    ValueError
        If the specification is not compatible with the Zarr format.
    """
    if zarr_format == 2:
        # Zarr v2 doesn't support ChunkGrid instances
        if isinstance(chunks, ChunkGrid):
            raise ValueError(
                "ChunkGrid instances are only supported in Zarr format 3. "
                "For Zarr format 2, use a tuple of integers for chunks."
            )

        # Zarr v2 doesn't support nested sequences (variable chunks)
        if _is_nested_sequence(chunks):
            raise ValueError(
                "Variable chunks (nested sequences) are only supported in Zarr format 3. "
                "Use zarr_format=3 or provide a regular tuple for chunks."
            )

        # Zarr v2 doesn't support sharding
        if shards is not None:
            raise ValueError(
                f"Sharding is only supported in Zarr format 3. "
                f"Got zarr_format={zarr_format} with shards={shards}."
            )


def _validate_sharding_compatibility(
    chunks: Any,
    shards: Any,
) -> None:
    """
    Validate that chunk specification is compatible with sharding.

    Parameters
    ----------
    chunks : Any
        The chunks specification.
    shards : Any
        The shards specification.

    Raises
    ------
    ValueError
        If the chunk specification is not compatible with sharding.
    """
    if shards is not None:
        # ChunkGrid instances can't be used with sharding
        if isinstance(chunks, ChunkGrid):
            raise ValueError(
                "Cannot use ChunkGrid instances with sharding. "
                "When shards parameter is provided, chunks must be a tuple of integers "
                "or 'auto'."
            )

        # Variable chunks (nested sequences) can't be used with sharding
        if _is_nested_sequence(chunks):
            raise ValueError(
                "Cannot use variable chunks (nested sequences) with sharding. "
                "Sharding requires uniform chunk sizes."
            )


def _validate_data_compatibility(
    chunk_grid: ChunkGrid | None,
    has_data: bool,
) -> None:
    """
    Validate that chunk grid is compatible with creating from data.

    Parameters
    ----------
    chunk_grid : ChunkGrid | None
        The chunk grid.
    has_data : bool
        Whether the array is being created from existing data.

    Raises
    ------
    ValueError
        If the chunk grid is not compatible with from_array.
    """
    if has_data and chunk_grid is not None and isinstance(chunk_grid, RectilinearChunkGrid):
        # RectilinearChunkGrid doesn't work with from_array
        raise ValueError(
            "Cannot use RectilinearChunkGrid (variable-sized chunks) when creating array "
            "from data. "
            "The from_array function requires uniform chunk sizes. "
            "Use regular chunks instead, or create an empty array first and write data "
            "separately."
        )


def resolve_chunk_spec(
    *,
    chunks: ChunksLike,
    shards: ShardsLike | None,
    shape: tuple[int, ...],
    dtype_itemsize: int,
    zarr_format: int,
    has_data: bool = False,
) -> ResolvedChunkSpec:
    """
    Resolve chunk specification into a ChunkGrid and shards parameters.

    This function centralizes all chunk grid creation logic and error handling.
    It converts any chunk specification format into a concrete ChunkGrid instance
    and validates compatibility with:
    - Zarr format version (v2 vs v3)
    - Sharding requirements
    - Data source requirements (from_array vs init_array)

    Parameters
    ----------
    chunks : ChunksLike
        The chunks specification from the user. Can be:
        - A ChunkGrid instance (Zarr v3 only)
        - A nested sequence for variable-sized chunks (Zarr v3 only)
        - A tuple of integers for uniform chunks
        - A single integer (applied to all dimensions)
        - The literal "auto"
    shards : ShardsLike | None
        The shards specification from the user. When provided, chunks represents
        the inner chunk size within each shard, and shards represents the outer shard size.
    shape : tuple[int, ...]
        The array shape. Required for auto-chunking and validation.
    dtype_itemsize : int
        The item size of the dtype in bytes. Used for auto-chunking heuristics.
    zarr_format : {2, 3}
        The Zarr format version.
    has_data : bool, default=False
        Whether the array is being created from existing data. If True,
        RectilinearChunkGrid (variable chunks) will raise an error since
        from_array requires uniform chunks.

    Returns
    -------
    ResolvedChunkSpec
        A dataclass containing the resolved chunk_grid and shards.
        The chunk_grid is always a concrete ChunkGrid instance.

    Raises
    ------
    ValueError
        If the chunk specification is invalid for the given zarr_format,
        or if incompatible options are specified (e.g., RectilinearChunkGrid + shards,
        ChunkGrid + Zarr v2, variable chunks + sharding).
    TypeError
        If the chunks parameter has an invalid type.

    Examples
    --------
    >>> # Regular chunks, no sharding
    >>> spec = resolve_chunk_spec(
    ...     chunks=(10, 10),
    ...     shards=None,
    ...     shape=(100, 100),
    ...     dtype_itemsize=4,
    ...     zarr_format=3
    ... )
    >>> spec.chunk_grid.chunk_shape
    (10, 10)
    >>> spec.shards is None
    True

    >>> # Sharding enabled
    >>> spec = resolve_chunk_spec(
    ...     chunks=(5, 5),
    ...     shards=(20, 20),
    ...     shape=(100, 100),
    ...     dtype_itemsize=4,
    ...     zarr_format=3
    ... )
    >>> spec.chunk_grid.chunk_shape  # Inner chunks
    (5, 5)
    >>> spec.shards  # Outer shards
    (20, 20)

    >>> # Variable chunks (RectilinearChunkGrid)
    >>> spec = resolve_chunk_spec(
    ...     chunks=[[10, 20, 30], [25, 25, 25, 25]],
    ...     shards=None,
    ...     shape=(60, 100),
    ...     dtype_itemsize=4,
    ...     zarr_format=3
    ... )
    >>> isinstance(spec.chunk_grid, RectilinearChunkGrid)
    True

    >>> # Error: variable chunks with Zarr v2
    >>> try:
    ...     resolve_chunk_spec(
    ...         chunks=[[10, 20], [5, 5]],
    ...         shards=None,
    ...         shape=(30, 10),
    ...         dtype_itemsize=4,
    ...         zarr_format=2
    ...     )
    ... except ValueError as e:
    ...     print(str(e))
    Variable chunks (nested sequences) are only supported in Zarr format 3...
    """
    # Step 1: Validate Zarr format compatibility
    _validate_zarr_format_compatibility(chunks, shards, zarr_format)

    # Step 2: Validate sharding compatibility
    _validate_sharding_compatibility(chunks, shards)

    # Step 3: Resolve the chunk specification to a ChunkGrid
    if shards is not None:
        # Sharding enabled: create ChunkGrid for inner chunks
        # Parse the inner chunks specification (must be regular, not variable)
        if isinstance(chunks, tuple):
            # Already normalized tuple
            inner_chunk_grid = RegularChunkGrid(chunk_shape=chunks)
        elif chunks == "auto":
            # Auto-chunk for inner chunks - use smaller target (1MB default for sharding)
            inner_chunks = _guess_chunks(shape, dtype_itemsize, max_bytes=1024 * 1024)
            inner_chunk_grid = RegularChunkGrid(chunk_shape=inner_chunks)
        elif isinstance(chunks, int):
            # Convert single int to tuple for all dimensions
            inner_chunks = _normalize_chunks(chunks, shape, dtype_itemsize)
            inner_chunk_grid = RegularChunkGrid(chunk_shape=inner_chunks)
        else:
            # This should have been caught by _validate_sharding_compatibility
            # but be defensive
            raise TypeError(
                f"Invalid chunks type when sharding is enabled: {type(chunks)}. "
                "Expected tuple, int, or 'auto'."
            )

        # Normalize shards to tuple[int, ...] for ResolvedChunkSpec
        shards_param: tuple[int, ...] | None
        if isinstance(shards, tuple):
            shards_param = shards
        elif isinstance(shards, dict):
            # ShardsConfigParam - extract the shape
            shards_param = shards.get("shape")
        else:
            # shards == "auto" or other cases
            # For "auto" shards, we pass None and let init_array handle it
            shards_param = None

        return ResolvedChunkSpec(
            chunk_grid=inner_chunk_grid,
            shards=shards_param,
        )
    else:
        # No sharding - use parse_chunk_grid to handle ChunkGrid, nested sequences, etc.
        chunk_grid = parse_chunk_grid(
            chunks, shape=shape, item_size=dtype_itemsize, zarr_format=zarr_format
        )

        # Step 4: Validate data compatibility
        _validate_data_compatibility(chunk_grid, has_data)

        # Step 5: Return the chunk_grid
        return ResolvedChunkSpec(
            chunk_grid=chunk_grid,
            shards=None,
        )
