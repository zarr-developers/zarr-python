"""
The `v2` chunk key encoding (Zarr v3 core spec).

The chunk with grid index `(k, j, i, ...)` is stored under the key
`k<sep>j<sep>i...`; the single chunk of a zero-dimensional array is stored
under `"0"`. The separator defaults to `"."`. This reproduces the chunk
layout of Zarr v2 stores, so existing v2 arrays can be converted to v3
without renaming chunks; it is not recommended for new arrays.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

from zarr_metadata.v3.chunk_key_encoding.v2 import (
    V2_CHUNK_KEY_ENCODING_NAME,
    V2ChunkKeyEncodingConfiguration,
    V2ChunkKeyEncodingName,
    V2ChunkKeyEncodingObject,
)

from zarr_chunk_key_encoding._abc import ChunkKeyEncoding, ChunkKeyEncodingJSON
from zarr_chunk_key_encoding._errors import ChunkKeyDecodeError
from zarr_chunk_key_encoding._parsing import (
    Separator,
    normalize_chunk_coords,
    parse_grid_index,
    parse_named_config_json,
    parse_separator,
)


@dataclass(frozen=True)
class V2ChunkKeyEncoding(ChunkKeyEncoding):
    """The v3 `v2` (v2-compatibility) chunk key encoding.

    Attributes
    ----------
    separator : Separator
        The character placed between coordinates. Defaults to `"."`.

    Examples
    --------
    >>> V2ChunkKeyEncoding().encode((1, 23))
    '1.23'
    >>> V2ChunkKeyEncoding(separator="/").decode("1/23")
    (1, 23)
    >>> V2ChunkKeyEncoding().encode(())
    '0'

    Notes
    -----
    This encoding is not injective at rank zero: the empty grid index `()`
    encodes to `"0"`, the same key produced by the rank-one index `(0,)`.
    `decode("0")` returns `(0,)`; recovering `()` requires knowing out
    of band that the array is zero-dimensional.
    """

    name: ClassVar[V2ChunkKeyEncodingName] = V2_CHUNK_KEY_ENCODING_NAME
    separator: Separator = "."

    def __post_init__(self) -> None:
        object.__setattr__(self, "separator", parse_separator(self.separator))

    @classmethod
    def from_json(cls, data: ChunkKeyEncodingJSON) -> Self:
        """Construct from `"v2"` or `{"name": "v2", ...}` metadata.

        Raises
        ------
        ChunkKeyConfigurationError
            If the metadata is not a valid description of this encoding.
        """
        configuration = parse_named_config_json(
            data,
            expected_name=cls.name,
            allowed_configuration_keys=("separator",),
        )
        if "separator" in configuration:
            return cls(separator=parse_separator(configuration["separator"]))
        return cls()

    def to_json(self) -> V2ChunkKeyEncodingObject:
        """Return the metadata object form, with the separator always explicit."""
        return V2ChunkKeyEncodingObject(
            name=self.name,
            configuration=V2ChunkKeyEncodingConfiguration(separator=self.separator),
        )

    def encode(self, chunk_coords: Sequence[int]) -> str:
        """Encode chunk grid indices into a store key.

        Raises
        ------
        InvalidChunkCoordsError
            If the coordinates are not non-negative integers.
        """
        indices = normalize_chunk_coords(chunk_coords)
        if not indices:
            return "0"
        return self.separator.join(map(str, indices))

    def decode(self, chunk_key: str) -> tuple[int, ...]:
        """Decode a store key into chunk grid indices.

        Inverse of `encode` for arrays of rank one and higher; see the
        class docstring for the rank-zero ambiguity.

        Raises
        ------
        ChunkKeyDecodeError
            If the key is not a valid output of `encode` for this separator.
        """
        if chunk_key == "":
            raise ChunkKeyDecodeError(
                f"Invalid chunk key '' for the {self.name!r} chunk key encoding."
            )
        parts = chunk_key.split(self.separator)
        return tuple(parse_grid_index(part, chunk_key) for part in parts)
