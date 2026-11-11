"""Strategies for each codec this package defines.

A codec handed an array draws what fits the chunk: a `bytes` codec an
`endian` for a type whose values take several bytes, a `transpose` an
`order` with an entry for each axis, a shard inner chunks that divide
it. Handed `Chunk()`, nothing is known of the chunk, and what is drawn
is what the codec's rules accept. The declarations a codec holds are drawn
from the strategies: the data type a `cast_value` casts to, and a
shard's two pipelines, each drawn for the chunk the shard's definition
says it is handed.
"""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Any, Final

from hypothesis import reject
from hypothesis import strategies as st

from zarr_metadata.definition._definition import AcceptedDeclaration, EmptyConfiguration
from zarr_metadata.testing._data_type import (
    COMPLEX_WIDTHS,
    FLOAT_WIDTHS,
    INTEGER_RANGES,
    float_fill_values,
    is_zero_float,
)
from zarr_metadata.testing._draw import (
    briefly,
    data_type_declarations,
    data_type_names,
    fill_values_of,
    pipelines_for,
    read,
)
from zarr_metadata.testing._strategy import EMPTY_CONFIGURATIONS, MAX_RANK, CodecStrategy
from zarr_metadata.v3._definition import Chunk, DataTypeDefinition, storage_of
from zarr_metadata.v3.codec._arithmetic import FLOATING_POINT, NO_REAL_NUMBERS, NOT_NUMBERS
from zarr_metadata.v3.codec.blosc import (
    BLOSC_CNAME,
    BLOSC_CODEC,
    BLOSC_NO_SHUFFLE,
    BLOSC_SHUFFLE,
    BloscCodecConfiguration,
)
from zarr_metadata.v3.codec.bytes import BYTES_CODEC, ENDIANNESS, BytesCodecConfiguration
from zarr_metadata.v3.codec.cast_value import (
    CAST_OUT_OF_RANGE_MODE,
    CAST_ROUNDING_MODE,
    CAST_VALUE_CODEC,
    CastOutOfRangeMode,
    CastValueCodecConfiguration,
    ScalarMap,
)
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC
from zarr_metadata.v3.codec.gzip import GZIP_CODEC, GzipCodecConfiguration
from zarr_metadata.v3.codec.scale_offset import SCALE_OFFSET_CODEC, ScaleOffsetCodecConfiguration
from zarr_metadata.v3.codec.sharding_indexed import (
    SHARDING_INDEX_LOCATION,
    SHARDING_INDEXED_CODEC,
    ShardingIndexedCodecConfiguration,
)
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC, TransposeCodecConfiguration
from zarr_metadata.v3.codec.vlen_bytes import VLEN_BYTES_CODEC
from zarr_metadata.v3.codec.vlen_utf8 import VLEN_UTF8_CODEC
from zarr_metadata.v3.codec.zstd import (
    ZSTD_CODEC,
    ZSTD_MAX_LEVEL,
    ZSTD_MIN_LEVEL,
    ZstdCodecConfiguration,
)
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_timedelta64 import NUMPY_TIMEDELTA64_DATA_TYPE

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata._common import JSONValue
    from zarr_metadata.definition._definition import ResolvedDeclaration
    from zarr_metadata.testing._strategy import Configurations, Strategies
    from zarr_metadata.v3._definition import CodecDefinition, CodecSize, Lengths

_BLOSC_MAX_TYPESIZE: Final = 255
"""The widest element c-blosc takes: `BLOSC_MAX_TYPESIZE`."""

_BLOSC_MAX_BLOCKSIZE: Final = 2**31 - 1
"""The largest block c-blosc takes, a signed 32-bit size; 0 lets it choose."""

_MAX_DIVISOR_SEARCH: Final = 2**12
"""The largest divisor of a shard's length searched for: one of an axis longer than its square is still found by its pair."""

_MAX_INNER_LENGTH: Final = 16
"""The longest inner chunk drawn along an axis whose shard lengths are unknown."""

_MAX_SHARD_DEPTH: Final = 1
"""How deep shards are drawn inside a shard: one inside another, and none inside that."""

_MAX_SCALARS: Final = 3
"""The most entries each direction of a `cast_value` `scalar_map` is drawn with."""

_STATIC: Final[frozenset[CodecSize]] = frozenset({"static"})
"""What a shard index's codecs are: of static size, so a reader finds the index before reading it."""

_ENDIAN: Final = st.sampled_from(ENDIANNESS)

_SMALL_NUMBERS: Final = st.integers(0, 127)
"""Values every integral and floating-point data type holds: for a chunk whose data type is not known."""

_NUMPY_TIMES: Final = frozenset({NUMPY_DATETIME64_DATA_TYPE, NUMPY_TIMEDELTA64_DATA_TYPE})
"""The numpy time types, each stored as a signed 64-bit integer."""

_NUMPY_TIME_RANGE: Final = (-(2**63) + 1, 2**63 - 1)
"""The signed 64-bit integers but the least, which numpy reads as `NaT`: the finite numpy times."""


@st.composite
def _endians(draw: st.DrawFn, *, required: bool) -> BytesCodecConfiguration:
    configuration = BytesCodecConfiguration()
    if required or draw(st.booleans()):
        configuration["endian"] = draw(_ENDIAN)
    return configuration


def _bytes(
    chunk: Chunk, _strategies: Strategies
) -> st.SearchStrategy[BytesCodecConfiguration] | None:
    """An `endian`, required for a data type whose values take several bytes; None for one whose values vary in size."""
    storage = None if chunk.data_type is None else storage_of(chunk.data_type)
    if storage == "variable_length":
        return None
    return _endians(required=storage == "multi_byte")


@st.composite
def _blosc(draw: st.DrawFn) -> BloscCodecConfiguration:
    """A compressor, a level, a shuffle and a block size; and a `typesize`, which only `noshuffle` may leave out.

    Under `noshuffle` the spec says `typesize` "is ignored", so it is
    drawn there too, a spelling that is not the simplest.
    """
    shuffle = draw(st.sampled_from(BLOSC_SHUFFLE))
    configuration = BloscCodecConfiguration(
        cname=draw(st.sampled_from(BLOSC_CNAME)),
        clevel=draw(st.integers(0, 9)),
        shuffle=shuffle,
        blocksize=draw(st.integers(0, _BLOSC_MAX_BLOCKSIZE)),
    )
    if shuffle != BLOSC_NO_SHUFFLE or draw(st.booleans()):
        configuration["typesize"] = draw(st.integers(1, _BLOSC_MAX_TYPESIZE))
    return configuration


def _order(rank: int) -> st.SearchStrategy[TransposeCodecConfiguration]:
    return st.permutations(range(rank)).map(
        lambda order: TransposeCodecConfiguration(order=tuple(order))
    )


def _transpose(
    chunk: Chunk, _strategies: Strategies
) -> st.SearchStrategy[TransposeCodecConfiguration]:
    """A permutation of the chunk's axes; of some number of axes, when the chunk does not say."""
    if chunk.rank is not None:
        return _order(chunk.rank)
    return st.integers(0, MAX_RANK).flatmap(_order)


@st.composite
def _offsets_and_scales(
    draw: st.DrawFn, offsets: st.SearchStrategy[JSONValue], scales: st.SearchStrategy[JSONValue]
) -> ScaleOffsetCodecConfiguration:
    configuration = ScaleOffsetCodecConfiguration()
    if draw(st.booleans()):
        configuration["offset"] = draw(offsets)
    if draw(st.booleans()):
        configuration["scale"] = draw(scales)
    return configuration


def _scale_offset(
    chunk: Chunk, _strategies: Strategies
) -> st.SearchStrategy[ScaleOffsetCodecConfiguration] | None:
    """An `offset` and a `scale`, each a value of the data type the codec is handed, the scale not zero; None for a data type whose values are not numbers.

    Each is a fill value of the chunk's data type
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L39-L43),
    so a float's is in every spelling a fill value takes, a `NaN` and an
    infinity among them, and a complex number's is a pair of them. The
    scale is drawn non-zero in any spelling, so that decoding never divides
    by zero; a scale small enough that decoding overflows is still drawn. A
    numpy time is drawn as the signed 64-bit integer it is stored as, never
    `NaT`, since it "is compatible with any codec that supports arrays of
    signed 64-bit integers"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L120).
    A data type this package does not define, whose values it cannot
    draw, takes only the configuration with neither, the identity.
    Handed a chunk whose data type is unknown, the values are small
    non-negative integers, which every integer and float type holds.
    """
    data_type = chunk.data_type
    definition = data_type.definition if isinstance(data_type, AcceptedDeclaration) else None
    if definition is None:
        offset: st.SearchStrategy[JSONValue] = _SMALL_NUMBERS
        scale: st.SearchStrategy[JSONValue] = st.integers(1, 127)
    elif definition.name in NOT_NUMBERS:
        return None
    elif definition in INTEGER_RANGES or definition in _NUMPY_TIMES:
        low, high = INTEGER_RANGES.get(definition, _NUMPY_TIME_RANGE)
        offset = st.integers(low, high)
        scale = st.integers(low, high).filter(lambda value: value != 0)
    elif definition in FLOAT_WIDTHS:
        offset = float_fill_values(FLOAT_WIDTHS[definition])
        scale = offset.filter(lambda value: not is_zero_float(value))
    elif definition in COMPLEX_WIDTHS:
        part = float_fill_values(COMPLEX_WIDTHS[definition])
        offset = st.tuples(part, part)
        # Non-zero in either part is a non-zero complex number.
        scale = st.tuples(part, part).filter(
            lambda pair: not (is_zero_float(pair[0]) and is_zero_float(pair[1]))
        )
    else:
        return st.builds(ScaleOffsetCodecConfiguration)
    return _offsets_and_scales(offset, scale)


@st.composite
def _zstd(draw: st.DrawFn) -> ZstdCodecConfiguration:
    """A level, and a checksum or none, which means no checksum."""
    configuration = ZstdCodecConfiguration(level=draw(st.integers(ZSTD_MIN_LEVEL, ZSTD_MAX_LEVEL)))
    if draw(st.booleans()):
        configuration["checksum"] = draw(st.booleans())
    return configuration


def _vlen(definition: CodecDefinition[Any]) -> CodecStrategy:
    """A codec with nothing to configure that takes only some data types: none for a chunk of another, which its own chunk rules say."""
    return CodecStrategy(
        configurations=lambda chunk, _strategies: (
            None
            if any(definition.chunk_rules(EmptyConfiguration(), {}, chunk))
            else EMPTY_CONFIGURATIONS
        )
    )


def _unchunked(configurations: Configurations) -> CodecStrategy:
    """A codec whose configurations do not depend on the chunk it is handed."""
    return CodecStrategy(configurations=lambda _chunk, _strategies: configurations)


def _name_read(data_type: ResolvedDeclaration[DataTypeDefinition[Any]] | None) -> str | None:
    """The name of the definition that accepted `data_type`; None when none did."""
    return data_type.definition.name if isinstance(data_type, AcceptedDeclaration) else None


def _source_scalars(chunk: Chunk, strategies: Strategies) -> st.SearchStrategy[JSONValue]:
    """Values of the chunk's data type: its fill values, or small numbers when no scope read it or no strategy draws for it."""
    source = chunk.data_type
    if not isinstance(source, AcceptedDeclaration) or source.definition not in strategies:
        return _SMALL_NUMBERS
    return fill_values_of(source, strategies)


@st.composite
def _scalar_maps(
    draw: st.DrawFn, sources: st.SearchStrategy[JSONValue], targets: st.SearchStrategy[JSONValue]
) -> ScalarMap:
    """Entries pairing a value of each type, the one an entry maps from first."""
    scalar_map = ScalarMap()
    if draw(st.booleans()):
        scalar_map["encode"] = tuple(
            draw(st.lists(st.tuples(sources, targets), max_size=_MAX_SCALARS))
        )
    if draw(st.booleans()):
        scalar_map["decode"] = tuple(
            draw(st.lists(st.tuples(targets, sources), max_size=_MAX_SCALARS))
        )
    return scalar_map


@st.composite
def _cast_value_configuration(
    draw: st.DrawFn,
    targets: list[str],
    sources: st.SearchStrategy[JSONValue],
    strategies: Strategies,
) -> CastValueCodecConfiguration:
    """A data type to cast to, a rounding, an out-of-range mode it allows, and a scalar map between the two types."""
    declaration = draw(data_type_declarations(targets, strategies))
    target: ResolvedDeclaration[DataTypeDefinition[Any]] = read(
        declaration, DataTypeDefinition, strategies
    )
    modes: tuple[CastOutOfRangeMode, ...] = CAST_OUT_OF_RANGE_MODE
    if _name_read(target) in FLOATING_POINT:
        modes = tuple(mode for mode in modes if mode != "wrap")
    configuration = CastValueCodecConfiguration(data_type=declaration)
    if draw(st.booleans()):
        configuration["rounding"] = draw(st.sampled_from(CAST_ROUNDING_MODE))
    if draw(st.booleans()):
        configuration["out_of_range"] = draw(st.sampled_from(modes))
    if draw(st.booleans()):
        configuration["scalar_map"] = draw(
            _scalar_maps(sources, fill_values_of(target, strategies))
        )
    return configuration


def _cast_value(
    chunk: Chunk, strategies: Strategies
) -> st.SearchStrategy[CastValueCodecConfiguration] | None:
    """A cast to a data type that models real numbers, from one that does; None for a chunk of one that does not.

    The codec "is only defined for data types that model real numbers"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L5-L9),
    on either side of the cast. Each `scalar_map` entry pairs a value of
    the type it casts from with one of the type it casts to, in the order
    of its direction; `wrap` is drawn only for a type that is not
    floating-point.
    """
    if _name_read(chunk.data_type) in NO_REAL_NUMBERS:
        return None
    targets = [name for name in data_type_names(strategies) if name not in NO_REAL_NUMBERS]
    if len(targets) == 0:
        return None
    return _cast_value_configuration(targets, _source_scalars(chunk, strategies), strategies)


def _divisors(lengths: frozenset[int] | None) -> st.SearchStrategy[int]:
    """An inner chunk length that divides each of `lengths`; any, when they are unknown or none but 0."""
    common = 0 if lengths is None else math.gcd(*lengths)
    if common == 0:
        return st.integers(1, _MAX_INNER_LENGTH)
    # Divisors come in pairs, one of each no greater than the square root.
    small = [
        n for n in range(1, min(math.isqrt(common), _MAX_DIVISOR_SEARCH) + 1) if common % n == 0
    ]
    # 1 and the length itself go last: Hypothesis draws the first of several
    # options far more often when a draw of variable length comes before it.
    return st.sampled_from(
        sorted({*small, *(common // n for n in small)}, key=lambda n: (n in (1, common), n))
    )


def _inner_chunk_shapes(lengths: Lengths | None) -> st.SearchStrategy[tuple[int, ...]]:
    """An inner chunk length for each axis of the shard, dividing the lengths shards take along it."""
    if lengths is None:
        return st.lists(st.integers(1, _MAX_INNER_LENGTH), max_size=MAX_RANK).map(tuple)
    return st.tuples(*(_divisors(axis) for axis in lengths))


@st.composite
def _sharding_configuration(
    draw: st.DrawFn, chunk: Chunk, strategies: Strategies, inner: Strategies
) -> ShardingIndexedCodecConfiguration:
    """Inner chunks that divide the shard, and the two pipelines, each drawn for what the shard's definition says it is handed.

    The inner codecs are drawn from `inner`, which says how a shard
    inside this one is drawn, if at all.
    """
    chunk_shape = draw(_inner_chunk_shapes(chunk.lengths))
    sized: ShardingIndexedCodecConfiguration = {
        "chunk_shape": chunk_shape,
        "codecs": (),
        "index_codecs": (),
    }
    handed = SHARDING_INDEXED_CODEC.pipelines(sized, {}, chunk)
    codecs = pipelines_for(handed["codecs"], inner)
    index_codecs = pipelines_for(handed["index_codecs"], strategies, _STATIC)
    if codecs is None or index_codecs is None:
        reject()  # a codec that takes a chunk of the data type, but not of these lengths
    configuration = ShardingIndexedCodecConfiguration(
        chunk_shape=chunk_shape, codecs=draw(codecs), index_codecs=draw(index_codecs)
    )
    if draw(st.booleans()):
        configuration["index_location"] = draw(st.sampled_from(SHARDING_INDEX_LOCATION))
    return configuration


_NO_SHAPE: Final[ShardingIndexedCodecConfiguration] = {
    "chunk_shape": (),
    "codecs": (),
    "index_codecs": (),
}
"""A shard configuration that says nothing but what its definition's `pipelines` reads: the data type each pipeline is handed."""


def _sharding(
    chunk: Chunk, strategies: Strategies, *, depth: int
) -> st.SearchStrategy[ShardingIndexedCodecConfiguration] | None:
    """A shard of the chunk, holding shards `depth` deep at most; None when no pipeline is drawn for its inner chunks or its index.

    Each pipeline needs a codec that holds no pipelines and takes the
    data type its definition hands it -- the shard's own for the inner
    chunks, and for the index one of static size; whether one does is
    asked of a chunk of that data type and nothing else known, since the
    inner lengths are drawn after. Its inner codecs are drawn from the
    same strategies but for a shard, which holds one level fewer, or is
    not drawn: shards drawn without end would nest without end. So is any
    definition named as a shard, which would otherwise claim the name in the
    scope the inner codecs are read in.
    """
    handed = SHARDING_INDEXED_CODEC.pipelines(_NO_SHAPE, {}, Chunk(None, chunk.data_type))
    if pipelines_for(Chunk(None, handed["codecs"].data_type), strategies) is None:
        return None
    if pipelines_for(Chunk(None, handed["index_codecs"].data_type), strategies, _STATIC) is None:
        return None
    inner = {d: s for d, s in strategies.items() if d.name != SHARDING_INDEXED_CODEC.name}
    if depth > 0:
        inner[SHARDING_INDEXED_CODEC] = _shards(depth - 1)
    return _sharding_configuration(chunk, strategies, briefly(inner))


def _shards(depth: int) -> CodecStrategy:
    """Shards that hold shards `depth` deep at most."""
    return CodecStrategy(configurations=partial(_sharding, depth=depth))


CODEC_STRATEGIES: Final[Mapping[CodecDefinition[Any], CodecStrategy]] = {
    BLOSC_CODEC: _unchunked(_blosc()),
    BYTES_CODEC: CodecStrategy(configurations=_bytes),
    CAST_VALUE_CODEC: CodecStrategy(configurations=_cast_value),
    CRC32C_CODEC: _unchunked(EMPTY_CONFIGURATIONS),
    GZIP_CODEC: _unchunked(
        st.integers(0, 9).map(lambda level: GzipCodecConfiguration(level=level))
    ),
    SCALE_OFFSET_CODEC: CodecStrategy(configurations=_scale_offset),
    SHARDING_INDEXED_CODEC: _shards(_MAX_SHARD_DEPTH),
    TRANSPOSE_CODEC: CodecStrategy(configurations=_transpose),
    VLEN_BYTES_CODEC: _vlen(VLEN_BYTES_CODEC),
    VLEN_UTF8_CODEC: _vlen(VLEN_UTF8_CODEC),
    ZSTD_CODEC: _unchunked(_zstd()),
}
"""A strategy for each codec."""

__all__ = ["CODEC_STRATEGIES"]
