"""Upgrades that read invalid stored array metadata documents written by older software.

This is the only place invalid metadata is read leniently; the metadata constructors
are strict. An upgrade maps a stored array metadata document (parsed JSON) to a valid
one and warns once when it changes anything. `ArrayV2Metadata.from_dict` and
`ArrayV3Metadata.from_dict` apply the upgrades for their Zarr format, so every path
that parses a stored document, including consolidated metadata, goes through them.

To read another kind of invalid document, add an upgrade to `V2_ARRAY_UPGRADES` or
`V3_ARRAY_UPGRADES`.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from typing_extensions import TypeIs

from zarr.core.chunk_grids import full_span_chunk_size
from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from zarr.core.common import JSON

type ArrayDocument = Mapping[str, JSON]
type Upgrade = Callable[[ArrayDocument], ArrayDocument]

RESAVE_HINT: Final = (
    "To store valid metadata, open the array writable and call `array.update_attributes({})`; "
    "if a group holds consolidated metadata for the array, then also call "
    "`zarr.consolidate_metadata` on that group."
)


def _warn(message: str) -> None:
    # The synchronous API parses metadata on zarr's IO thread, whose stack holds no
    # user code, so the warning points at the upgrade on every path.
    warnings.warn(f"{message} {RESAVE_HINT}", ZarrUserWarning, stacklevel=2)


def _is_int_list(value: object) -> TypeIs[list[int] | tuple[int, ...]]:
    """Whether `value` is a JSON array of integers (JSON `false` and `true` count)."""
    return isinstance(value, list | tuple) and all(isinstance(v, int) for v in value)


def _read_invalid_chunk_sizes(chunk_shape: JSON, shape: JSON, unit: JSON) -> list[int] | None:
    """Read a regular chunk shape whose chunk sizes include 0, JSON `false` or JSON `true`.

    0 and `false` are read as one chunk spanning the axis, a multiple of `unit` on that
    axis (the inner chunk shape of a sharded array); `true` is read as 1. Returns `None`
    when there is nothing to read this way: anything else that is invalid is left for
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
    _warn(message)
    return upgraded


def _invalid_chunk_sizes_v2(doc: ArrayDocument) -> ArrayDocument:
    chunks = _read_invalid_chunk_sizes(doc.get("chunks"), doc.get("shape"), None)
    return doc if chunks is None else {**doc, "chunks": chunks}


def _sharding_chunk_shape(codecs: JSON) -> JSON:
    """The inner chunk shape of a sharding codec in a Zarr format 3 codec list, if any."""
    if isinstance(codecs, Sequence) and not isinstance(codecs, str):
        for codec in codecs:
            if isinstance(codec, Mapping) and codec.get("name") == "sharding_indexed":
                configuration = codec.get("configuration")
                if isinstance(configuration, Mapping):
                    return configuration.get("chunk_shape")
    return None


def _invalid_chunk_sizes_v3(doc: ArrayDocument) -> ArrayDocument:
    grid = doc.get("chunk_grid")
    if not isinstance(grid, Mapping) or grid.get("name") != "regular":
        return doc
    configuration = grid.get("configuration")
    if not isinstance(configuration, Mapping):
        return doc
    chunk_shape = _read_invalid_chunk_sizes(
        configuration.get("chunk_shape"),
        doc.get("shape"),
        _sharding_chunk_shape(doc.get("codecs")),
    )
    if chunk_shape is None:
        return doc
    return {
        **doc,
        "chunk_grid": {**grid, "configuration": {**configuration, "chunk_shape": chunk_shape}},
    }


V2_ARRAY_UPGRADES: Final[tuple[Upgrade, ...]] = (_invalid_chunk_sizes_v2,)
V3_ARRAY_UPGRADES: Final[tuple[Upgrade, ...]] = (_invalid_chunk_sizes_v3,)


def upgrade_array_document(doc: ArrayDocument, upgrades: Sequence[Upgrade]) -> ArrayDocument:
    """Apply `upgrades` to a stored array metadata document, in order."""
    for upgrade in upgrades:
        doc = upgrade(doc)
    return doc
