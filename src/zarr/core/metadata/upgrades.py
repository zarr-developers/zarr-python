"""Upgrades that read invalid stored array metadata documents written by older software.

This is the only place invalid metadata is read leniently; the metadata constructors
are strict. An upgrade maps a stored array metadata document (parsed JSON) to a valid
one. `ArrayV2Metadata.from_dict` and `ArrayV3Metadata.from_dict` apply the upgrades for
their Zarr format, so every path that parses a stored document, including consolidated
metadata, goes through them.

A reading warns only where the user must act on it; a reading that gives what zarr
read from the same document before these upgrades existed is silent, so a document
that opened without a warning still does. The warnings are given once per document,
after the upgraded document has passed the metadata constructor, so an invalid
document raises its own error, not a warning about how it was read. Silent or not,
metadata read from an upgraded document is marked (see `mark_upgraded`), so the array
stores the upgrade before it writes chunks under it.

To read another kind of invalid document, add an upgrade to `ARRAY_UPGRADES`.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import chain, repeat
from typing import TYPE_CHECKING, Final, TypeGuard

from zarr.core._json import json_equal
from zarr.core.chunk_grids import full_span_chunk_size
from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

type ArrayDocument = Mapping[str, JSON]
type Upgrade = Callable[[ArrayDocument], tuple[ArrayDocument, str | None] | None]
"""Returns `None` if the document needs no upgrade, else the upgraded document and,
if the user must act on how it was read, a warning saying so (else `None`)."""

RESAVE_HINT: Final = (
    "To store valid metadata, open the array writable and call `array.update_attributes({})`; "
    "if a group holds consolidated metadata for the array, then also call "
    "`zarr.consolidate_metadata` on that group."
)


def mark_upgraded[M](metadata: M, readings: Sequence[str | None], path: str | None) -> M:
    """Record that `metadata` was read from a stored document that needed the upgrades
    whose `readings` `upgrade_array_document` returned, if any: set
    `_stored_document_upgraded` on `metadata`, so the array stores the upgrade before
    it writes chunks under it, and warn once with the readings that are warnings,
    naming the array at `path` when the caller knows it."""
    if readings:
        object.__setattr__(metadata, "_stored_document_upgraded", True)
    if messages := [reading for reading in readings if reading is not None]:
        subject = "" if path is None else f"Array {path!r}: "
        # The synchronous API parses metadata on zarr's IO thread, whose stack holds no
        # user code, so the warning points at the `from_dict` that read the document.
        warnings.warn(f"{subject}{' '.join(messages)} {RESAVE_HINT}", ZarrUserWarning, stacklevel=2)
    return metadata


def _is_int_list(value: object) -> TypeGuard[list[int]]:
    """Whether `value` is a JSON array of integers (JSON `false` and `true` count)."""
    return isinstance(value, list) and all(isinstance(v, int) for v in value)


def _abbreviate(value: JSON, limit: int = 60) -> str:
    """`value` as JSON, cut to at most `limit` characters."""
    text = json.dumps(value)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _read_chunk_size(
    size: JSON, span: int | None, unit: int
) -> tuple[int | list[JSON], str | None] | None:
    """Read one entry of a stored regular chunk shape as a chunk edge length, or as the
    chunk edge lengths of its axis.

    Returns the reading and, where the user must act on how it was read, how it was
    read; `None` if the entry cannot be read, which leaves it for the metadata
    constructors to check. A JSON int >= 1 is kept, JSON `true` is read as 1, and 0 or
    JSON `false` is read as one chunk spanning the axis of length `span`, a multiple of
    `unit` (the inner chunk size of a shard): on an axis of positive length no chunk
    can have been stored under it, so the array holds only its fill value. `span` is
    `None` where no stored 0 is known, as in the inner chunk shape of a sharding codec:
    0 is then left as stored. A flat JSON list is kept as the chunk edge lengths of its
    axis, which only a rectilinear chunk grid can declare (see `_invalid_chunk_sizes_v3`);
    its edges are read as those of a stored rectilinear chunk grid (see
    `_invalid_edge_lengths_v3`).
    """
    match size:
        case True:
            return 1, None
        case int() if size >= 1:
            return size, None
        case int() if size == 0 and span is not None:
            edge = full_span_chunk_size(span, unit)
            if span == 0:
                return edge, None
            return edge, (
                f"one chunk spanning the dimension ({edge}), and as no chunk can be stored "
                "under a chunk size of 0, the array holds only its fill value"
            )
        case list() if not any(isinstance(edge, list) for edge in size):
            return size, None
    return None


def _read_chunk_shape(
    stored: JSON, spans: Sequence[int | None], units: Iterable[int] = ()
) -> tuple[list[int | list[JSON]], str | None] | None:
    """Read a stored regular chunk shape, entry by entry (see `_read_chunk_size`), for
    axes of lengths `spans` whose chunks are multiples of `units` (1 where not given).

    Returns the chunk shape and, where the user must act on how an entry was read, a
    sentence saying how the chunk shape was read; `None` if it cannot be read.
    """
    if not (isinstance(stored, list) and len(stored) == len(spans)):
        return None
    edges: list[int | list[JSON]] = []
    readings: list[str] = []
    axes = zip(stored, spans, chain(units, repeat(1)), strict=False)
    for axis, (size, span, unit) in enumerate(axes):
        read = _read_chunk_size(size, span, unit)
        if read is None:
            return None
        edge, how = read
        edges.append(edge)
        if how is not None:
            readings.append(f"{json.dumps(size)} in dimension {axis} as {how}")
    if not readings:
        return edges, None
    return edges, (
        f"The stored chunk shape {_abbreviate(stored)} is invalid: chunk sizes must be "
        f"integers of at least 1. It is read as {_abbreviate(edges)}, reading "
        f"{'; '.join(readings)}."
    )


def _invalid_chunk_sizes_v2(doc: ArrayDocument) -> tuple[ArrayDocument, str | None] | None:
    shape = doc.get("shape")
    if not _is_int_list(shape):
        return None
    stored = doc.get("chunks")
    match _read_chunk_shape(stored, shape):
        case chunks, reading if not json_equal(chunks, stored):
            return {**doc, "chunks": chunks}, reading
    return None


def _read_codec(codec: JSON) -> JSON:
    """Read a stored codec: the inner chunk shape of a sharding codec is read as a chunk
    shape with no known axis lengths (see `_read_chunk_shape`), and so are those of the
    sharding codecs nested in its codecs."""
    if not (isinstance(codec, Mapping) and codec.get("name") == "sharding_indexed"):
        return codec
    configuration = codec.get("configuration")
    if not isinstance(configuration, Mapping):
        return codec
    upgraded = dict(configuration)
    stored = configuration.get("chunk_shape")
    if isinstance(stored, list):
        match _read_chunk_shape(stored, [None] * len(stored)):
            case chunk_shape, _:
                upgraded["chunk_shape"] = list(chunk_shape)
    if isinstance(codecs := configuration.get("codecs"), list):
        upgraded["codecs"] = [_read_codec(inner) for inner in codecs]
    return {**codec, "configuration": upgraded}


def _invalid_inner_chunk_sizes_v3(doc: ArrayDocument) -> tuple[ArrayDocument, str | None] | None:
    stored = doc.get("codecs")
    if not isinstance(stored, list):
        return None
    codecs = [_read_codec(codec) for codec in stored]
    if json_equal(codecs, stored):
        return None
    return {**doc, "codecs": codecs}, None


def _inner_chunk_shape(doc: ArrayDocument) -> list[int]:
    """The inner chunk shape of the sharding codec of a Zarr format 3 array document, if
    it has one and that is a list of integers; else `[]`."""
    codecs = doc.get("codecs")
    if isinstance(codecs, list):
        for codec in codecs:
            match codec:
                case {
                    "name": "sharding_indexed",
                    "configuration": {"chunk_shape": list() as inner},
                }:
                    return inner if _is_int_list(inner) else []
    return []


def _invalid_chunk_sizes_v3(doc: ArrayDocument) -> tuple[ArrayDocument, str | None] | None:
    grid = doc.get("chunk_grid")
    shape = doc.get("shape")
    if not (isinstance(grid, Mapping) and grid.get("name") == "regular" and _is_int_list(shape)):
        return None
    configuration = grid.get("configuration")
    if not isinstance(configuration, Mapping):
        return None
    stored = configuration.get("chunk_shape")
    read = _read_chunk_shape(stored, shape, _inner_chunk_shape(doc))
    if read is None:
        return None
    chunk_shape, reading = read
    edge_axes = [axis for axis, size in enumerate(chunk_shape) if isinstance(size, list)]
    if not edge_axes:
        if json_equal(chunk_shape, stored):
            return None
        upgraded = {**configuration, "chunk_shape": chunk_shape}
        return {**doc, "chunk_grid": {**grid, "configuration": upgraded}}, reading
    if len(edge_axes) == len(chunk_shape):
        # Only a mix of chunk sizes and edge lists was ever stored in a regular grid.
        return None
    # The user must act: re-saving this grid needs the rectilinear chunks flag.
    as_rectilinear = (
        f"The stored chunk grid is named 'regular', but its chunk shape lists chunk edge "
        f"lengths in dimensions {edge_axes}, which only a rectilinear chunk grid can declare. "
        "It is read as that rectilinear chunk grid. Re-saving the metadata stores that "
        "rectilinear chunk grid, so each step that follows requires "
        "`zarr.config.set({'array.rectilinear_chunks': True})`."
    )
    rectilinear: JSON = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": chunk_shape},
    }
    return {**doc, "chunk_grid": rectilinear}, " ".join(filter(None, (reading, as_rectilinear)))


def _read_edge_length(edge: JSON) -> JSON:
    """Read one stored chunk edge length of a rectilinear chunk grid: JSON `true` is
    read as 1 and an integral JSON float of at least 1 (`4.0`) as the `int` it equals.
    Anything else is kept, for the metadata constructors to check."""
    match edge:
        case True:
            return 1
        case float() if edge.is_integer() and edge >= 1:
            return int(edge)
    return edge


def _read_rectilinear_axis(spec: JSON) -> JSON:
    """Read the stored chunk edge lengths of one axis of a rectilinear chunk grid (see
    `_read_edge_length`): its explicit edges and the sizes of its run-length encoded
    `[size, count]` pairs. A bare chunk size and a repeat count are kept as stored: no
    writer stored them as `true` or as floats."""
    match spec:
        case list():
            return [_read_rectilinear_entry(entry) for entry in spec]
    return spec


def _read_rectilinear_entry(entry: JSON) -> JSON:
    match entry:
        case [size, count]:
            return [_read_edge_length(size), count]
    return _read_edge_length(entry)


def _invalid_edge_lengths_v3(doc: ArrayDocument) -> tuple[ArrayDocument, str | None] | None:
    grid = doc.get("chunk_grid")
    if not (isinstance(grid, Mapping) and grid.get("name") == "rectilinear"):
        return None
    configuration = grid.get("configuration")
    if not isinstance(configuration, Mapping):
        return None
    stored = configuration.get("chunk_shapes")
    if not isinstance(stored, list):
        return None
    read = [_read_rectilinear_axis(axis) for axis in stored]
    if json_equal(read, stored):
        return None
    upgraded = {**configuration, "chunk_shapes": read}
    return {**doc, "chunk_grid": {**grid, "configuration": upgraded}}, None


ARRAY_UPGRADES: Final[Mapping[ZarrFormat, tuple[Upgrade, ...]]] = {
    2: (_invalid_chunk_sizes_v2,),
    # The inner chunk shape is read first: it gives the unit of the outer chunk shape.
    # The rectilinear edge lengths are read last, after any upgrade that yields a
    # rectilinear chunk grid.
    3: (_invalid_inner_chunk_sizes_v3, _invalid_chunk_sizes_v3, _invalid_edge_lengths_v3),
}
"""The upgrades of an array document of each Zarr format, applied in order."""


def upgrade_array_document(
    doc: ArrayDocument, zarr_format: ZarrFormat
) -> tuple[ArrayDocument, list[str | None]]:
    """Apply the upgrades for `zarr_format` to a stored array metadata document.

    Returns the upgraded document and the reading of each upgrade that changed it (a
    warning, or `None` for a silent one), for `mark_upgraded` once the document has
    been validated.
    """
    readings: list[str | None] = []
    for upgrade in ARRAY_UPGRADES[zarr_format]:
        upgraded = upgrade(doc)
        if upgraded is not None:
            doc, reading = upgraded
            readings.append(reading)
    return doc, readings
