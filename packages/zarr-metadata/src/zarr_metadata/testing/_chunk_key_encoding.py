"""Strategies for each chunk key encoding this package defines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from hypothesis import strategies as st

from zarr_metadata.testing._strategy import ChunkKeyEncodingStrategy
from zarr_metadata.v3.chunk_key_encoding.default import (
    DEFAULT_CHUNK_KEY_ENCODING,
    DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR,
)
from zarr_metadata.v3.chunk_key_encoding.v2 import (
    V2_CHUNK_KEY_ENCODING,
    V2_CHUNK_KEY_ENCODING_SEPARATOR,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.v3._definition import ChunkKeyEncodingDefinition

CHUNK_KEY_ENCODING_STRATEGIES: Final[
    Mapping[ChunkKeyEncodingDefinition[Any], ChunkKeyEncodingStrategy]
] = {
    DEFAULT_CHUNK_KEY_ENCODING: ChunkKeyEncodingStrategy(
        configurations=st.fixed_dictionaries(
            {}, optional={"separator": st.sampled_from(DEFAULT_CHUNK_KEY_ENCODING_SEPARATOR)}
        )
    ),
    V2_CHUNK_KEY_ENCODING: ChunkKeyEncodingStrategy(
        configurations=st.fixed_dictionaries(
            {}, optional={"separator": st.sampled_from(V2_CHUNK_KEY_ENCODING_SEPARATOR)}
        )
    ),
}
"""A strategy for each chunk key encoding: a `separator`, or none, which means the encoding's default."""

__all__ = ["CHUNK_KEY_ENCODING_STRATEGIES"]
