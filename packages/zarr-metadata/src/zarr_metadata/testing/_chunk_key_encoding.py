"""Strategies for each chunk key encoding this package defines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from hypothesis import strategies as st

from zarr_metadata.testing._strategy import ChunkKeyEncodingStrategy
from zarr_metadata.v3.chunk_key_encoding.default import (
    DEFAULT_CHUNK_KEY_ENCODING,
    DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR,
    DefaultChunkKeyEncodingConfiguration,
)
from zarr_metadata.v3.chunk_key_encoding.v2 import (
    V2_CHUNK_KEY_ENCODING,
    V2_CHUNK_KEY_ENCODING_SEPARATOR,
    V2ChunkKeyEncodingConfiguration,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.v3._definition import ChunkKeyEncodingDefinition

_DEFAULT_SEPARATORS: Final = st.builds(DefaultChunkKeyEncodingConfiguration) | st.sampled_from(
    DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR
).map(lambda separator: DefaultChunkKeyEncodingConfiguration(separator=separator))

_V2_SEPARATORS: Final = st.builds(V2ChunkKeyEncodingConfiguration) | st.sampled_from(
    V2_CHUNK_KEY_ENCODING_SEPARATOR
).map(lambda separator: V2ChunkKeyEncodingConfiguration(separator=separator))

CHUNK_KEY_ENCODING_STRATEGIES: Final[
    Mapping[ChunkKeyEncodingDefinition[Any], ChunkKeyEncodingStrategy]
] = {
    DEFAULT_CHUNK_KEY_ENCODING: ChunkKeyEncodingStrategy(
        configurations=lambda _strategies: _DEFAULT_SEPARATORS
    ),
    V2_CHUNK_KEY_ENCODING: ChunkKeyEncodingStrategy(
        configurations=lambda _strategies: _V2_SEPARATORS
    ),
}
"""A strategy for each chunk key encoding: a `separator`, or none, which means the encoding's default."""

__all__ = ["CHUNK_KEY_ENCODING_STRATEGIES"]
