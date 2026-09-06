"""
The `default` chunk key encoding (Zarr v3 core spec).

The chunk with grid index `(k, j, i, ...)` is stored under the key
`c<sep>k<sep>j<sep>i...`; the single chunk of a zero-dimensional array is
stored under `"c"`. The separator defaults to `"/"`.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

from zarr_metadata.v3.chunk_key_encoding.default import (
    DEFAULT_CHUNK_KEY_ENCODING_NAME,
    DefaultChunkKeyEncodingConfiguration,
    DefaultChunkKeyEncodingName,
    DefaultChunkKeyEncodingObject,
)

from zarr_chunk_key_encoding._abc import ChunkKeyEncoding, ChunkKeyEncodingJSON
from zarr_chunk_key_encoding._errors import ChunkKeyDecodeError
from zarr_chunk_key_encoding._parsing import (
    Separator,
    _parse_separator,
    normalize_chunk_coords,
    parse_grid_index,
    parse_named_config_json,
)


@dataclass(frozen=True)
class DefaultChunkKeyEncoding(ChunkKeyEncoding):
    """The v3 `default` chunk key encoding.

    Attributes
    ----------
    separator : Separator
        The character placed between the `c` prefix and each coordinate.
        Defaults to `"/"`.

    Examples
    --------
    >>> DefaultChunkKeyEncoding().encode((1, 23))
    'c/1/23'
    >>> DefaultChunkKeyEncoding(separator=".").decode("c.1.23")
    (1, 23)
    >>> DefaultChunkKeyEncoding().encode(())
    'c'
    """

    name: ClassVar[DefaultChunkKeyEncodingName] = DEFAULT_CHUNK_KEY_ENCODING_NAME
    separator: Separator = "/"

    def __post_init__(self) -> None:
        object.__setattr__(self, "separator", _parse_separator(self.separator))

    @classmethod
    def from_json(cls, data: ChunkKeyEncodingJSON) -> Self:
        """Construct from `"default"` or `{"name": "default", ...}` metadata.

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
            return cls(separator=_parse_separator(configuration["separator"]))
        return cls()

    def to_json(self) -> DefaultChunkKeyEncodingObject:
        """Return the metadata object form, with the separator always explicit."""
        return DefaultChunkKeyEncodingObject(
            name=self.name,
            configuration=DefaultChunkKeyEncodingConfiguration(separator=self.separator),
        )

    def encode(self, chunk_coords: Sequence[int]) -> str:
        """Encode chunk grid indices into a store key.

        Raises
        ------
        InvalidChunkCoordsError
            If the coordinates are not non-negative integers.
        """
        indices = normalize_chunk_coords(chunk_coords)
        return self.separator.join(("c", *map(str, indices)))

    def decode(self, chunk_key: str) -> tuple[int, ...]:
        """Decode a store key into chunk grid indices. Exact inverse of `encode`.

        Raises
        ------
        ChunkKeyDecodeError
            If the key is not a valid output of `encode` for this separator.
        """
        if chunk_key == "c":
            return ()
        prefix = "c" + self.separator
        if not chunk_key.startswith(prefix):
            raise ChunkKeyDecodeError(
                f"Invalid chunk key {chunk_key!r} for the {self.name!r} chunk key "
                f"encoding: expected {prefix!r} followed by "
                f"{self.separator!r}-separated coordinates, or exactly 'c'."
            )
        parts = chunk_key[len(prefix) :].split(self.separator)
        return tuple(parse_grid_index(part, chunk_key) for part in parts)
