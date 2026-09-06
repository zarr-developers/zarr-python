"""
Internal parsing helpers shared by the concrete chunk key encodings.

This module is private. It holds the strict validators for the pieces every
encoding needs: the v3 named-configuration JSON envelope, chunk grid indices
supplied to `encode`, and the textual coordinate parts consumed by `decode`.
"""

import operator
from collections.abc import Mapping, Sequence
from typing import Final, Literal, cast

from zarr_metadata import JSONValue

from zarr_chunk_key_encoding._errors import (
    ChunkKeyConfigurationError,
    ChunkKeyDecodeError,
    InvalidChunkCoordsError,
)

Separator = Literal[".", "/"]
"""Literal type of the permitted chunk key separators."""

_SEPARATORS: Final = (".", "/")

# Keys permitted in the top-level named-configuration envelope, per the v3
# core spec.
_ENVELOPE_KEYS: Final = frozenset({"name", "configuration", "must_understand"})


def parse_separator(data: object) -> Separator:
    """Validate and narrow a chunk key separator."""
    if data not in _SEPARATORS:
        raise ChunkKeyConfigurationError(
            f"Invalid chunk key separator: {data!r}. Expected one of {_SEPARATORS}."
        )
    return data


def parse_named_config_json(
    data: object,
    *,
    expected_name: str,
    allowed_configuration_keys: tuple[str, ...],
) -> Mapping[str, JSONValue]:
    """Validate a named-configuration JSON envelope and return its configuration.

    Accepts either the spec's short-hand name string or the object form
    `{"name": ..., "configuration": {...}, "must_understand": ...}`.
    Validation is strict: unexpected envelope keys, a mismatched `name`, a
    non-mapping `configuration`, unexpected configuration keys, and a
    `must_understand` that is not the boolean `true` are all rejected.

    `must_understand: false` is rejected rather than ignored. The v3 spec
    does not support it for this extension point -- an implementation that
    meets a chunk key encoding it does not recognize cannot skip the array,
    it has to fail -- so the field is meaningless here and a document
    carrying it is malformed. `true` is accepted as a redundant spelling of
    the default.

    Parameters
    ----------
    data : object
        Unvalidated JSON: the short-hand name string or the object form.
    expected_name : str
        The registered name the `name` field must equal.
    allowed_configuration_keys : tuple of str
        The set of keys permitted in `configuration`.

    Returns
    -------
    Mapping[str, JSONValue]
        The (possibly empty) configuration mapping.

    Raises
    ------
    ChunkKeyConfigurationError
        If the envelope is malformed in any of the ways described above.
    """
    if isinstance(data, str):
        if data != expected_name:
            raise ChunkKeyConfigurationError(
                f"Invalid chunk key encoding name: {data!r}. Expected {expected_name!r}."
            )
        return {}
    if not isinstance(data, Mapping):
        raise ChunkKeyConfigurationError(
            f"Invalid chunk key encoding metadata: expected the name string "
            f"{expected_name!r} or a JSON object, got {data!r}."
        )
    # Decoded JSON objects always have string keys; assert that view for the
    # type checker. A caller passing a hand-built mapping need not honour
    # that, so non-string keys are reported by the extra-keys check below --
    # sorted by `repr`, since sorting a mix of, say, `1` and `"zz"` directly
    # raises TypeError, which would escape this package's error hierarchy.
    mapping = cast("Mapping[str, JSONValue]", data)

    extra_keys = sorted((k for k in mapping if k not in _ENVELOPE_KEYS), key=repr)
    if extra_keys:
        raise ChunkKeyConfigurationError(
            f"Invalid chunk key encoding metadata: unexpected keys {extra_keys}. "
            f"Permitted keys: {sorted(_ENVELOPE_KEYS)}."
        )
    if "name" not in mapping:
        raise ChunkKeyConfigurationError(
            f"Invalid chunk key encoding metadata: missing required key 'name' in {data!r}."
        )
    if mapping["name"] != expected_name:
        raise ChunkKeyConfigurationError(
            f"Invalid chunk key encoding name: {mapping['name']!r}. Expected {expected_name!r}."
        )
    if "must_understand" in mapping and mapping["must_understand"] is not True:
        raise ChunkKeyConfigurationError(
            f"Invalid chunk key encoding metadata: 'must_understand' must be "
            f"true, got {mapping['must_understand']!r}. The Zarr v3 spec does "
            f"not support 'must_understand': false for chunk key encodings, "
            f"since a reader that does not recognize one cannot skip it."
        )

    configuration = mapping.get("configuration", {})
    if not isinstance(configuration, Mapping):
        raise ChunkKeyConfigurationError(
            f"Invalid chunk key encoding metadata: 'configuration' must be a "
            f"JSON object, got {configuration!r}."
        )
    extra_config_keys = sorted(
        (k for k in configuration if k not in allowed_configuration_keys), key=repr
    )
    if extra_config_keys:
        raise ChunkKeyConfigurationError(
            f"Invalid configuration for chunk key encoding {expected_name!r}: "
            f"unexpected keys {extra_config_keys}. "
            f"Permitted keys: {sorted(allowed_configuration_keys)}."
        )
    return configuration


def normalize_chunk_coords(chunk_coords: Sequence[int]) -> tuple[int, ...]:
    """Normalize chunk grid indices to a tuple of built-in non-negative ints.

    Accepts any objects implementing `__index__` (so NumPy integers work),
    which also normalizes `True`/`False` to `1`/`0` instead of letting
    them stringify as `"True"`/`"False"`.

    Parameters
    ----------
    chunk_coords : Sequence[int]
        The chunk grid indices to normalize.

    Returns
    -------
    tuple of int
        The normalized indices.

    Raises
    ------
    InvalidChunkCoordsError
        If any coordinate is not an integer or is negative.
    """
    try:
        indices = tuple(operator.index(c) for c in chunk_coords)
    except TypeError as e:
        raise InvalidChunkCoordsError(
            f"Chunk coordinates must be integers. Got {chunk_coords!r}."
        ) from e
    for index in indices:
        if index < 0:
            raise InvalidChunkCoordsError(
                f"Chunk coordinates must be non-negative. Got {chunk_coords!r}."
            )
    return indices


def parse_grid_index(part: str, chunk_key: str) -> int:
    """Parse one coordinate substring of a chunk key into a grid index.

    Only canonical decimal representations are accepted: ASCII digits with no
    sign and no leading zeros (except `"0"` itself). This guarantees that
    decoding is the exact inverse of encoding; keys like `"c/01"` were not
    produced by these encodings and are rejected rather than silently
    normalized.

    Parameters
    ----------
    part : str
        One separator-delimited component of the chunk key.
    chunk_key : str
        The full chunk key, for error messages.

    Returns
    -------
    int
        The parsed grid index.

    Raises
    ------
    ChunkKeyDecodeError
        If the component is not a canonical non-negative decimal integer.
    """
    if not (part.isascii() and part.isdigit()) or (len(part) > 1 and part[0] == "0"):
        raise ChunkKeyDecodeError(
            f"Invalid chunk key {chunk_key!r}: {part!r} is not a canonical non-negative integer."
        )
    try:
        return int(part)
    except ValueError as e:
        raise ChunkKeyDecodeError(
            f"Invalid chunk key {chunk_key!r}: {part!r} is not a canonical non-negative integer."
        ) from e
