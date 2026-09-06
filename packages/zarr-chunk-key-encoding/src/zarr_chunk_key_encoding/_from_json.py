"""Construct core chunk key encodings from Zarr v3 JSON metadata."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, cast

from zarr_chunk_key_encoding._abc import ChunkKeyEncoding, ChunkKeyEncodingJSON
from zarr_chunk_key_encoding._default import DefaultChunkKeyEncoding
from zarr_chunk_key_encoding._errors import (
    ChunkKeyConfigurationError,
    UnknownChunkKeyEncodingError,
)
from zarr_chunk_key_encoding._v2 import V2ChunkKeyEncoding

if TYPE_CHECKING:
    from zarr_metadata import JSONValue

_CHUNK_KEY_ENCODINGS: Final[Mapping[str, type[ChunkKeyEncoding]]] = {
    DefaultChunkKeyEncoding.name: DefaultChunkKeyEncoding,
    V2ChunkKeyEncoding.name: V2ChunkKeyEncoding,
}


def _get_chunk_key_encoding_class(name: str) -> type[ChunkKeyEncoding]:
    """Return the core encoding class registered under a metadata name."""
    try:
        return _CHUNK_KEY_ENCODINGS[name]
    except KeyError:
        raise UnknownChunkKeyEncodingError(name, tuple(_CHUNK_KEY_ENCODINGS)) from None


def chunk_key_encoding_from_json(data: ChunkKeyEncodingJSON) -> ChunkKeyEncoding:
    """Construct a core chunk key encoding from Zarr v3 JSON metadata.

    Parameters
    ----------
    data : ChunkKeyEncodingJSON
        The short-hand name string or named-configuration object form.

    Returns
    -------
    ChunkKeyEncoding
        The constructed encoding.

    Raises
    ------
    ChunkKeyConfigurationError
        If the metadata carries no usable `name`, or the named class rejects
        the metadata.
    UnknownChunkKeyEncodingError
        If the name is not one of the two encodings defined by the Zarr v3
        core specification.
    """
    data_obj = cast("object", data)
    if isinstance(data_obj, str):
        name = data_obj
    elif isinstance(data_obj, Mapping):
        name = cast("Mapping[str, JSONValue]", data_obj).get("name")
        if not isinstance(name, str):
            raise ChunkKeyConfigurationError(
                f"Invalid chunk key encoding metadata: expected a 'name' key "
                f"with a string value in {data!r}."
            )
    else:
        raise ChunkKeyConfigurationError(
            f"Invalid chunk key encoding metadata: expected a name string or "
            f"a JSON object, got {data!r}."
        )
    return _get_chunk_key_encoding_class(name).from_json(data)
