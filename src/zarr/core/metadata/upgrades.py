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
document raises its own error, not a warning about how it was read. Metadata read
from a document whose upgrade moves chunks (a chunk size read as another size) is
marked (see `mark_upgraded`), silent or not, so the array stores the upgrade before it
writes chunks under it. An upgrade that only respells a value zarr already read the
same way (`true` as 1, `4.0` as 4) moves no chunks, so it is not marked.

To read another kind of invalid document, add an upgrade to `ARRAY_UPGRADES`.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import chain, repeat
from typing import TYPE_CHECKING, Final, NamedTuple, TypeGuard, cast

from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

type ArrayDocument = Mapping[str, JSON]


class Reading(NamedTuple):
    """How an upgrade read a stored document."""

    moves_chunks: bool
    """Whether chunks written under the upgraded document are not where a reader of the
    stored document looks for them, so the array must store the upgrade before it
    writes chunks."""
    warning: str | None
    """What the user must act on, if anything."""


type Upgrade = Callable[[ArrayDocument], tuple[ArrayDocument, Reading] | None]
"""Returns `None` if the document needs no upgrade, else the upgraded document and how
it was read."""

RESAVE_HINT: Final = (
    "To store valid metadata, open the array writable and call `array.update_attributes({})`; "
    "if a group holds consolidated metadata for the array, then also call "
    "`zarr.consolidate_metadata` on that group."
)
"""How to store the upgrade of a document whose array holds data."""

RECREATE_HINT: Final = (
    "The array holds only its fill value, so recreate it with the chunk shape you want; "
    "nothing is lost. To keep it instead, store valid metadata: open the array writable and "
    "call `array.update_attributes({})`; if a group holds consolidated metadata for the "
    "array, then also call `zarr.consolidate_metadata` on that group."
)
"""How to act on a document whose array can hold no data: a stored chunk size of 0 is read
as the smallest chunk size, which is a poor chunk shape for the data the array grows into,
and a re-save would keep it."""


def mark_upgraded[M](
    metadata: M,
    stored: ArrayDocument,
    readings: Sequence[Reading],
    path: str | None,
    *,
    warn: bool = True,
) -> M:
    """Record that `metadata` was read from the document `stored`, which needed the
    upgrades whose `readings` `upgrade_array_document` returned, if any. If a reading
    moves chunks, keep a copy of `stored` as `_stored_document` on `metadata`, so the
    array stores the upgrade before it writes chunks under it and consolidated metadata
    stores it as it was stored. If `warn`, warn once with the readings' warnings (each
    says what the user must act on and how), naming the array at `path` when the caller
    knows it."""
    if any(reading.moves_chunks for reading in readings):
        object.__setattr__(metadata, "_stored_document", copy.deepcopy(stored))
    messages = [reading.warning for reading in readings if reading.warning is not None]
    if warn and messages:
        subject = "" if path is None else f"Array {path!r}: "
        # The synchronous API parses metadata on zarr's IO thread, whose stack holds no
        # user code, so the warning points at the `from_dict` that read the document.
        warnings.warn(f"{subject}{' '.join(messages)}", ZarrUserWarning, stacklevel=2)
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
) -> tuple[int | list[JSON], bool, str | None] | None:
    """Read one entry of a stored regular chunk shape as a chunk edge length, or as the
    chunk edge lengths of its axis.

    Returns the reading, whether it moves chunks (it is not the size a reader of the
    stored entry uses), and, where the user must act on how it was read, how it was
    read; `None` if the entry cannot be read, which leaves it for the metadata
    constructors to check. A JSON int >= 1 is kept, JSON `true` is read as 1, and 0 or
    JSON `false` is read as `unit`, the smallest chunk edge length the axis can have (1,
    or the inner chunk size of a shard), whatever the length `span` of the axis. That is
    what `chunks=-1` gives on an axis of length 0, where these sizes were written, and it
    does not depend on how far the axis has grown since. On an axis of positive length no
    chunk can have been stored under a chunk size of 0, so the array holds only its fill
    value. `span` is `None` where no stored 0 is known, as in the inner chunk shape of a
    sharding codec: 0 is then left as stored. A flat JSON list is kept as the chunk edge
    lengths of its axis, which only a rectilinear chunk grid can declare (see
    `_invalid_chunk_sizes_v3`); its edges are read as those of a stored rectilinear chunk
    grid (see `_invalid_edge_lengths_v3`).
    """
    match size:
        case True:
            return 1, False, None
        case int() if size >= 1:
            return size, False, None
        case int() if size == 0 and span is not None:
            if span == 0:
                return unit, True, None
            return (
                unit,
                True,
                (
                    "1, as no chunk can be stored under a chunk size of 0"
                    if unit == 1
                    else f"{unit}, the inner chunk size, as no chunk can be stored under a "
                    "chunk size of 0"
                ),
            )
        case list() if not any(isinstance(edge, list) for edge in size):
            return size, False, None
    return None


def _read_chunk_shape(
    stored: JSON, spans: Sequence[int | None], units: Iterable[int] = ()
) -> tuple[list[int | list[JSON]], Reading | None] | None:
    """Read a stored regular chunk shape, entry by entry (see `_read_chunk_size`), for
    axes of lengths `spans` whose chunks are multiples of `units` (1 where not given).

    Returns the chunk shape and how it was read if any entry was read as another value
    (the warning says how the chunk shape was read and, as the array then holds only its
    fill value, recommends recreating it, see `RECREATE_HINT`), else `None`; `None` if
    it cannot be read.
    """
    if not (isinstance(stored, list) and len(stored) == len(spans)):
        return None
    edges: list[int | list[JSON]] = []
    changed = moves_chunks = False
    readings: list[str] = []
    axes = zip(stored, spans, chain(units, repeat(1)), strict=False)
    for axis, (size, span, unit) in enumerate(axes):
        read = _read_chunk_size(size, span, unit)
        if read is None:
            return None
        edge, moved, how = read
        edges.append(edge)
        changed |= size is True or edge != size
        moves_chunks |= moved
        if how is not None:
            readings.append(f"{json.dumps(size)} in dimension {axis} as {how}")
    if not changed:
        return edges, None
    warning = (
        f"The stored chunk shape {_abbreviate(stored)} is invalid: chunk sizes must be "
        f"integers of at least 1. It is read as {_abbreviate(edges)}, reading "
        f"{'; '.join(readings)}. {RECREATE_HINT}"
        if readings
        else None
    )
    return edges, Reading(moves_chunks, warning)


def _invalid_chunk_sizes_v2(doc: ArrayDocument) -> tuple[ArrayDocument, Reading] | None:
    shape = doc.get("shape")
    if not _is_int_list(shape):
        return None
    match _read_chunk_shape(doc.get("chunks"), shape):
        case chunks, Reading() as reading:
            return {**doc, "chunks": chunks}, reading
    return None


def _read_codec(codec: JSON) -> JSON | None:
    """Read a stored codec: the inner chunk shape of a sharding codec is read as a chunk
    shape with no known axis lengths (see `_read_chunk_shape`), and so are those of the
    sharding codecs nested in its codecs. Returns `None` if nothing was read as another
    value. No stored inner chunk size is read as a different size, so the reading moves
    no chunks."""
    match codec:
        case {"name": "sharding_indexed", "configuration": Mapping() as configuration}:
            upgraded: dict[str, JSON] = {}
            stored = configuration.get("chunk_shape")
            match (
                _read_chunk_shape(stored, [None] * len(stored))
                if isinstance(stored, list)
                else None
            ):
                case chunk_shape, Reading():
                    upgraded["chunk_shape"] = chunk_shape
            if isinstance(codecs := configuration.get("codecs"), list) and (
                inner := _read_codecs(codecs)
            ):
                upgraded["codecs"] = inner
            if not upgraded:
                return None
            # The mapping pattern does not narrow `codec` for mypy.
            return {
                **cast("Mapping[str, JSON]", codec),
                "configuration": {**configuration, **upgraded},
            }
    return None


def _read_codecs(codecs: list[JSON]) -> list[JSON] | None:
    """Read stored codecs (see `_read_codec`); `None` if nothing was read as another
    value."""
    read = [_read_codec(codec) for codec in codecs]
    if all(codec is None for codec in read):
        return None
    return [stored if new is None else new for stored, new in zip(codecs, read, strict=True)]


def _invalid_inner_chunk_sizes_v3(doc: ArrayDocument) -> tuple[ArrayDocument, Reading] | None:
    stored = doc.get("codecs")
    if not isinstance(stored, list) or (codecs := _read_codecs(stored)) is None:
        return None
    return {**doc, "codecs": codecs}, Reading(moves_chunks=False, warning=None)


def _inner_chunk_shape(doc: ArrayDocument) -> list[int] | None:
    """The inner chunk shape of the sharding codec of a Zarr format 3 array document:
    `[]` if it has none, `None` if its inner chunk sizes are not all integers of at
    least 1 (the unit of its outer chunk shape is then unknown)."""
    match doc.get("codecs"):
        case list() as codecs:
            for codec in codecs:
                match codec:
                    case {"name": "sharding_indexed", "configuration": {"chunk_shape": inner}}:
                        return inner if _is_int_list(inner) and min(inner, default=1) >= 1 else None
    return []


def _invalid_chunk_sizes_v3(doc: ArrayDocument) -> tuple[ArrayDocument, Reading] | None:
    grid = doc.get("chunk_grid")
    shape = doc.get("shape")
    if not (isinstance(grid, Mapping) and grid.get("name") == "regular" and _is_int_list(shape)):
        return None
    if (units := _inner_chunk_shape(doc)) is None:
        return None
    configuration = grid.get("configuration")
    if not isinstance(configuration, Mapping):
        return None
    read = _read_chunk_shape(configuration.get("chunk_shape"), shape, units)
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
    # The user must act: re-saving this grid needs the rectilinear chunks flag.
    as_rectilinear = (
        f"The stored chunk grid is named 'regular', but its chunk shape lists chunk edge "
        f"lengths in dimensions {edge_axes}, which only a rectilinear chunk grid can declare. "
        "It is read as that rectilinear chunk grid. Re-saving the metadata stores that "
        "rectilinear chunk grid, so each step that follows requires "
        f"`zarr.config.set({{'array.rectilinear_chunks': True}})`. {RESAVE_HINT}"
    )
    rectilinear: JSON = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": chunk_shape},
    }
    warning = " ".join(filter(None, (reading and reading.warning, as_rectilinear)))
    # Only zarr 3.2.x reads the stored grid, so the array stores the rectilinear grid
    # before it writes chunks, for every other reader to find them.
    return {**doc, "chunk_grid": rectilinear}, Reading(moves_chunks=True, warning=warning)


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


def _respelled(read: JSON, stored: JSON) -> bool:
    """Whether reading `stored` as `read` gave a value of another type anywhere in it
    (`true` read as 1, `4.0` as 4). Compares types, not encodings, so a value that is
    not JSON (a NumPy integer in metadata built in code) is simply kept."""
    if isinstance(stored, list) and isinstance(read, list):
        return any(_respelled(r, s) for r, s in zip(read, stored, strict=True))
    return type(read) is not type(stored)


def _invalid_edge_lengths_v3(doc: ArrayDocument) -> tuple[ArrayDocument, Reading] | None:
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
    if not _respelled(read, stored):
        return None
    upgraded = {**configuration, "chunk_shapes": read}
    # A stored edge is read as the value it equals, so the reading moves no chunks.
    return {**doc, "chunk_grid": {**grid, "configuration": upgraded}}, Reading(False, None)


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
) -> tuple[ArrayDocument, list[Reading]]:
    """Apply the upgrades for `zarr_format` to a stored array metadata document.

    Returns the upgraded document and the reading of each upgrade that changed it, for
    `mark_upgraded` once the document has been validated.
    """
    readings: list[Reading] = []
    for upgrade in ARRAY_UPGRADES[zarr_format]:
        upgraded = upgrade(doc)
        if upgraded is not None:
            doc, reading = upgraded
            readings.append(reading)
    return doc, readings
