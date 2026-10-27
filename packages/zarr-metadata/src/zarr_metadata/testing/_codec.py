"""Strategies for each codec this package defines that holds no other field.

A codec handed an array draws what fits the chunk: a `bytes` codec an
`endian` for a type whose values take several bytes, a `transpose` an
`order` with an entry for each axis. Handed `Chunk()`, nothing is known
of the chunk, and what is drawn is what the codec's rules accept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from hypothesis import strategies as st

from zarr_metadata.testing._data_type import FLOAT_WIDTHS, INTEGER_RANGES, finite_floats
from zarr_metadata.testing._strategy import CodecStrategy
from zarr_metadata.v3._definition import Chunk, storage_of
from zarr_metadata.v3.codec.blosc import BLOSC_CNAME, BLOSC_CODEC, BLOSC_NO_SHUFFLE, BLOSC_SHUFFLE
from zarr_metadata.v3.codec.bytes import BYTES_CODEC, ENDIANNESS
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.codec.scale_offset import SCALE_OFFSET_CODEC
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC, ZSTD_MAX_LEVEL, ZSTD_MIN_LEVEL

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata._common import JSONValue
    from zarr_metadata.v3._definition import CodecDefinition

_MAX_RANK: Final = 5
"""The most axes an `order` is drawn with, when the chunk does not say."""

_BLOSC_MAX_TYPESIZE: Final = 255
"""The widest element c-blosc takes: `BLOSC_MAX_TYPESIZE`."""

_BLOSC_MAX_BLOCKSIZE: Final = 2**31 - 1
"""The largest block c-blosc takes, a signed 32-bit size; 0 lets it choose."""

_ENDIAN: Final = st.sampled_from(ENDIANNESS)


def _bytes(chunk: Chunk) -> st.SearchStrategy[Mapping[str, JSONValue]] | None:
    """An `endian`, required for a data type whose values take several bytes; None for one whose values vary in size."""
    storage = None if chunk.data_type is None else storage_of(chunk.data_type)
    if storage == "variable_length":
        return None
    if storage == "multi_byte":
        return st.fixed_dictionaries({"endian": _ENDIAN})
    return st.fixed_dictionaries({}, optional={"endian": _ENDIAN})


@st.composite
def _blosc(draw: st.DrawFn) -> Mapping[str, JSONValue]:
    """A compressor, a level, a shuffle and a block size; and a `typesize`, which only `noshuffle` may leave out.

    Under `noshuffle` the spec says `typesize` "is ignored", so it is
    drawn there too, a spelling that is not the simplest.
    """
    shuffle = draw(st.sampled_from(BLOSC_SHUFFLE))
    typesize = st.integers(1, _BLOSC_MAX_TYPESIZE)
    required: dict[str, st.SearchStrategy[JSONValue]] = {
        "cname": st.sampled_from(BLOSC_CNAME),
        "clevel": st.integers(0, 9),
        "shuffle": st.just(shuffle),
        "blocksize": st.integers(0, _BLOSC_MAX_BLOCKSIZE),
    }
    if shuffle == BLOSC_NO_SHUFFLE:
        return draw(st.fixed_dictionaries(required, optional={"typesize": typesize}))
    return draw(st.fixed_dictionaries({**required, "typesize": typesize}))


def _order(rank: int) -> st.SearchStrategy[Mapping[str, JSONValue]]:
    return st.permutations(range(rank)).map(lambda order: {"order": tuple(order)})


def _transpose(chunk: Chunk) -> st.SearchStrategy[Mapping[str, JSONValue]]:
    """A permutation of the chunk's axes; of some number of axes, when the chunk does not say."""
    if chunk.rank is not None:
        return _order(chunk.rank)
    return st.integers(0, _MAX_RANK).flatmap(_order)


def _scale_offset(chunk: Chunk) -> st.SearchStrategy[Mapping[str, JSONValue]] | None:
    """An `offset` and a `scale`, each a finite value of the data type the codec is handed, the scale not zero.

    Each is a fill value of the chunk's data type
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L39-L43),
    drawn finite, and the scale non-zero, so that decoding, which divides
    by it, is defined. Handed a data type the codec's registry entry does
    not list -- truth values, raw bits, complex numbers, numpy times --
    nothing is drawn: for some the spec refuses it, and for the others
    leaves it open. Handed a chunk whose data type is unknown, the values
    are small non-negative integers, which every listed type holds.
    """
    data_type = chunk.data_type
    read = data_type is not None and data_type.resolution == "read"
    definition = data_type.definition if data_type is not None and read else None
    if definition is None:
        offset: st.SearchStrategy[JSONValue] = st.integers(0, 127)
        scale: st.SearchStrategy[JSONValue] = st.integers(1, 127)
    elif definition in INTEGER_RANGES:
        low, high = INTEGER_RANGES[definition]
        offset = st.integers(low, high)
        scale = st.integers(low, high).filter(lambda value: value != 0)
    elif definition in FLOAT_WIDTHS:
        offset = finite_floats(FLOAT_WIDTHS[definition])
        scale = finite_floats(FLOAT_WIDTHS[definition]).filter(lambda value: value != 0)
    else:
        return None
    return st.fixed_dictionaries({}, optional={"offset": offset, "scale": scale})


def _unchunked(configurations: st.SearchStrategy[Mapping[str, JSONValue]]) -> CodecStrategy:
    """A codec whose configurations do not depend on the chunk it is handed."""
    return CodecStrategy(configurations=lambda _chunk: configurations)


CODEC_STRATEGIES: Final[Mapping[CodecDefinition[Any], CodecStrategy]] = {
    BLOSC_CODEC: _unchunked(_blosc()),
    BYTES_CODEC: CodecStrategy(configurations=_bytes),
    CRC32C_CODEC: _unchunked(st.just({})),
    GZIP_CODEC: _unchunked(st.fixed_dictionaries({"level": st.integers(0, 9)})),
    SCALE_OFFSET_CODEC: CodecStrategy(configurations=_scale_offset),
    TRANSPOSE_CODEC: CodecStrategy(configurations=_transpose),
    ZSTD_CODEC: _unchunked(
        st.fixed_dictionaries(
            {"level": st.integers(ZSTD_MIN_LEVEL, ZSTD_MAX_LEVEL)},
            optional={"checksum": st.booleans()},
        )
    ),
}
"""A strategy for each codec that holds no other field."""

__all__ = ["CODEC_STRATEGIES"]
