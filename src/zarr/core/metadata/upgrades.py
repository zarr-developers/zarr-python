"""Upgrades that read invalid stored array metadata documents written by older software.

This is the only place invalid metadata is read leniently; the metadata constructors
are strict. An upgrade maps a stored array metadata document (parsed JSON) to a valid
one and says how it read the document. `ArrayV2Metadata.from_dict` and
`ArrayV3Metadata.from_dict` apply the upgrades for their Zarr format, so every path
that parses a stored document, including consolidated metadata, goes through them, and
warn once, with every reading, after the upgraded document has passed the metadata
constructor. An invalid document therefore raises its own error, not a warning about
how it was read.

To read another kind of invalid document, add an upgrade to `V2_ARRAY_UPGRADES` or
`V3_ARRAY_UPGRADES`.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import chain, repeat
from typing import TYPE_CHECKING, Final, TypeGuard, cast

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


def warn_readings(readings: Sequence[str], path: str | None) -> None:
    """Warn once with the readings returned by `upgrade_array_document`, naming the
    array at `path` when the caller knows it."""
    if readings:
        subject = "" if path is None else f"Array {path!r}: "
        # The synchronous API parses metadata on zarr's IO thread, whose stack holds no
        # user code, so the warning points at the `from_dict` that read the document.
        warnings.warn(f"{subject}{' '.join(readings)} {RESAVE_HINT}", ZarrUserWarning, stacklevel=2)


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

    Returns the reading and, for an invalid entry, how it was read; `None` if the
    entry cannot be read, which leaves it for the metadata constructors to reject. A
    JSON int >= 1 is kept, JSON `true` is read as 1, and 0 or JSON `false` is read as
    one chunk spanning the axis of length `span`, a multiple of `unit` (the inner chunk
    size of a shard). `span` is `None` where no stored 0 is known, as in the inner
    chunk shape of a sharding codec: 0 is then left for the constructors to reject. A
    flat JSON list is kept as the chunk edge lengths of its axis, which only a
    rectilinear chunk grid can declare (see `_invalid_chunk_sizes_v3`); the rectilinear
    chunk grid checks each edge.
    """
    match size:
        case True:
            return 1, "1"
        case int() if size >= 1:
            return size, None
        case int() if size == 0 and span is not None:
            edge = full_span_chunk_size(span, unit)
            how = f"one chunk spanning the dimension ({edge})"
            if span > 0:
                how += (
                    ", and as no chunk can be stored under a chunk size of 0, the array "
                    "holds only its fill value"
                )
            return edge, how
        case list() if not any(isinstance(edge, list) for edge in size):
            return size, None
    return None


def _read_chunk_shape(
    stored: JSON, spans: Sequence[int | None], units: Iterable[int], name: str
) -> tuple[list[int | list[JSON]], str | None] | None:
    """Read a stored regular chunk shape, entry by entry (see `_read_chunk_size`), for
    axes of lengths `spans` whose chunks are multiples of `units` (1 where not given).

    Returns the chunk shape and, if an entry is invalid, a sentence saying how the
    `name` was read; `None` if it cannot be read.
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
        f"The stored {name} {_abbreviate(stored)} is invalid: chunk sizes must be "
        f"integers of at least 1. It is read as {_abbreviate(edges)}, reading "
        f"{'; '.join(readings)}."
    )


def _invalid_chunk_sizes_v2(doc: ArrayDocument) -> tuple[ArrayDocument, str] | None:
    shape = doc.get("shape")
    if not _is_int_list(shape):
        return None
    match _read_chunk_shape(doc.get("chunks"), shape, (), "chunk shape"):
        case chunks, str(reading):
            return {**doc, "chunks": chunks}, reading
    return None


def _sharding_codec(doc: ArrayDocument) -> tuple[Sequence[JSON], int, Mapping[str, JSON]] | None:
    """The codec list of a Zarr format 3 array document, with the position and the
    configuration of its sharding codec, if it has one."""
    codecs = doc.get("codecs")
    if isinstance(codecs, list):
        for index, codec in enumerate(codecs):
            if isinstance(codec, Mapping) and codec.get("name") == "sharding_indexed":
                configuration = codec.get("configuration")
                if isinstance(configuration, Mapping):
                    return codecs, index, configuration
    return None


def _read_inner_chunk_shape(
    doc: ArrayDocument,
) -> tuple[Sequence[int], str | None] | None:
    """Read the inner chunk shape of a sharded array. No stored inner chunk size of 0
    or `false` is known, so the spans of its axes are not given."""
    shape = doc.get("shape")
    sharding = _sharding_codec(doc)
    if sharding is None or not _is_int_list(shape):
        return None
    _, _, configuration = sharding
    match _read_chunk_shape(
        configuration.get("chunk_shape"),
        [None] * len(shape),
        (),
        "inner chunk shape of the sharding codec",
    ):
        case inner, reading if _is_int_list(inner):
            return inner, reading
    return None


def _invalid_inner_chunk_sizes_v3(doc: ArrayDocument) -> tuple[ArrayDocument, str] | None:
    match _read_inner_chunk_shape(doc), _sharding_codec(doc):
        case (inner, str(reading)), (codecs, index, configuration):
            # `_sharding_codec` found a mapping at `index`.
            codec = cast("Mapping[str, JSON]", codecs[index])
            upgraded = {**codec, "configuration": {**configuration, "chunk_shape": inner}}
            return {**doc, "codecs": [*codecs[:index], upgraded, *codecs[index + 1 :]]}, reading
    return None


def _invalid_chunk_sizes_v3(doc: ArrayDocument) -> tuple[ArrayDocument, str] | None:
    grid = doc.get("chunk_grid")
    shape = doc.get("shape")
    if not (isinstance(grid, Mapping) and grid.get("name") == "regular" and _is_int_list(shape)):
        return None
    configuration = grid.get("configuration")
    if not isinstance(configuration, Mapping):
        return None
    inner = _read_inner_chunk_shape(doc)
    units = () if inner is None else inner[0]
    read = _read_chunk_shape(configuration.get("chunk_shape"), shape, units, "chunk shape")
    if read is None:
        return None
    chunk_shape, reading = read
    edge_axes = [axis for axis, size in enumerate(chunk_shape) if isinstance(size, list)]
    if not edge_axes:
        if reading is None:
            return None
        upgraded = {**configuration, "chunk_shape": chunk_shape}
        return {**doc, "chunk_grid": {**grid, "configuration": upgraded}}, reading
    if len(edge_axes) == len(chunk_shape):
        # Only a mix of chunk sizes and edge lists was ever stored in a regular grid.
        return None
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


V2_ARRAY_UPGRADES: Final[tuple[Upgrade, ...]] = (_invalid_chunk_sizes_v2,)
V3_ARRAY_UPGRADES: Final[tuple[Upgrade, ...]] = (
    _invalid_inner_chunk_sizes_v3,
    _invalid_chunk_sizes_v3,
)


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
