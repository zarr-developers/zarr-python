"""Upgrades that read invalid stored array metadata documents written by older software.

This is the only place invalid metadata is read leniently; the metadata constructors
are strict. An upgrade maps a stored array metadata document (parsed JSON) to a valid
one and says how it read the document. `ArrayV2Metadata.from_dict` and
`ArrayV3Metadata.from_dict` apply the upgrades for their Zarr format, so every path
that parses a stored document, including consolidated metadata, goes through them, and
warn with each reading once the upgraded document has passed the metadata constructor.
An invalid document therefore raises its own error, not a warning about how it was read.

To read another kind of invalid document, add an upgrade to `V2_ARRAY_UPGRADES` or
`V3_ARRAY_UPGRADES`.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from typing_extensions import TypeIs

from zarr.core.chunk_grids import full_span_chunk_size
from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from zarr.core.common import JSON

type ArrayDocument = Mapping[str, JSON]
type Upgrade = Callable[[ArrayDocument], tuple[ArrayDocument, str] | None]
"""Returns `None` if the document needs no upgrade, else the upgraded document and a
sentence saying how it was read."""

RESAVE_HINT: Final = (
    "To store valid metadata, open the array writable and call `array.update_attributes({})`; "
    "if a group holds consolidated metadata for the array, then also call "
    "`zarr.consolidate_metadata` on that group."
)


def warn_readings(readings: Iterable[str]) -> None:
    """Warn once for each reading returned by `upgrade_array_document`."""
    for reading in readings:
        # The synchronous API parses metadata on zarr's IO thread, whose stack holds no
        # user code, so the warning points at the `from_dict` that read the document.
        warnings.warn(f"{reading} {RESAVE_HINT}", ZarrUserWarning, stacklevel=2)


def _is_int_list(value: object) -> TypeIs[list[int] | tuple[int, ...]]:
    """Whether `value` is a JSON array of integers (JSON `false` and `true` count)."""
    return isinstance(value, list | tuple) and all(isinstance(v, int) for v in value)


def _read_invalid_chunk_sizes(
    chunk_shape: JSON, shape: JSON, unit: JSON
) -> tuple[list[int], str] | None:
    """Read a regular chunk shape whose chunk sizes include 0, JSON `false` or JSON `true`.

    0 and `false` are read as one chunk spanning the axis, a multiple of `unit` on that
    axis (the inner chunk shape of a sharded array); `true` is read as 1. Returns the
    upgraded chunk shape and how it was read, or `None` when there is nothing to read
    this way: anything else that is invalid is left for
    the metadata constructors to reject.
    """
    if not (_is_int_list(chunk_shape) and _is_int_list(shape)) or len(chunk_shape) != len(shape):
        return None
    invalid_axes = [
        axis for axis, size in enumerate(chunk_shape) if size == 0 or isinstance(size, bool)
    ]
    if not invalid_axes:
        return None
    units = (
        list(unit)
        if _is_int_list(unit) and len(unit) == len(shape) and all(u >= 1 for u in unit)
        else [1] * len(shape)
    )
    upgraded = [
        full_span_chunk_size(extent, int(u)) if size == 0 else int(size)
        for size, extent, u in zip(chunk_shape, shape, units, strict=True)
    ]
    readings = "; ".join(
        f"{json.dumps(chunk_shape[axis])} on axis {axis} as "
        + ("1" if chunk_shape[axis] else f"one chunk spanning the axis ({upgraded[axis]})")
        for axis in invalid_axes
    )
    message = (
        f"The stored chunk shape {json.dumps(list(chunk_shape))} is invalid: chunk sizes "
        f"must be integers of at least 1. It is read as {upgraded}, reading {readings}."
    )
    if any(chunk_shape[axis] == 0 and shape[axis] > 0 for axis in invalid_axes):
        message += (
            " No chunk can be stored under a chunk size of 0, so the array holds only its "
            "fill value."
        )
    return upgraded, message


def _invalid_chunk_sizes_v2(doc: ArrayDocument) -> tuple[ArrayDocument, str] | None:
    read = _read_invalid_chunk_sizes(doc.get("chunks"), doc.get("shape"), None)
    if read is None:
        return None
    chunks, reading = read
    return {**doc, "chunks": chunks}, reading


def _sharding_chunk_shape(codecs: JSON) -> JSON:
    """The inner chunk shape of a sharding codec in a Zarr format 3 codec list, if any."""
    if isinstance(codecs, Sequence) and not isinstance(codecs, str):
        for codec in codecs:
            if isinstance(codec, Mapping) and codec.get("name") == "sharding_indexed":
                configuration = codec.get("configuration")
                if isinstance(configuration, Mapping):
                    return configuration.get("chunk_shape")
    return None


def _invalid_chunk_sizes_v3(doc: ArrayDocument) -> tuple[ArrayDocument, str] | None:
    grid = doc.get("chunk_grid")
    if not isinstance(grid, Mapping) or grid.get("name") != "regular":
        return None
    configuration = grid.get("configuration")
    if not isinstance(configuration, Mapping):
        return None
    read = _read_invalid_chunk_sizes(
        configuration.get("chunk_shape"),
        doc.get("shape"),
        _sharding_chunk_shape(doc.get("codecs")),
    )
    if read is None:
        return None
    chunk_shape, reading = read
    return {
        **doc,
        "chunk_grid": {**grid, "configuration": {**configuration, "chunk_shape": chunk_shape}},
    }, reading


V2_ARRAY_UPGRADES: Final[tuple[Upgrade, ...]] = (_invalid_chunk_sizes_v2,)
V3_ARRAY_UPGRADES: Final[tuple[Upgrade, ...]] = (_invalid_chunk_sizes_v3,)


def upgrade_array_document(
    doc: ArrayDocument, upgrades: Sequence[Upgrade]
) -> tuple[ArrayDocument, list[str]]:
    """Apply `upgrades` to a stored array metadata document, in order.

    Returns the upgraded document and the readings of the upgrades that changed it,
    for `warn_readings` once the document has been validated.
    """
    readings: list[str] = []
    for upgrade in upgrades:
        upgraded = upgrade(doc)
        if upgraded is not None:
            doc, reading = upgraded
            readings.append(reading)
    return doc, readings
