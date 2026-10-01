"""Repairs that read invalid stored array metadata documents written by older software.

This is the only place invalid metadata is read leniently. A repair maps a stored
array metadata document (parsed JSON) to a valid one. `ArrayV2Metadata.from_dict` and
`ArrayV3Metadata.from_dict` apply the repairs for their Zarr format, so every path
that parses a stored document, including consolidated metadata, goes through them.

A reading warns only where the user must act on it; a reading that gives what zarr
read from the same document before these repairs existed is silent, so a document
that opened without a warning still does. The warnings are given once per document,
after the repaired document has passed the metadata constructor, so an invalid
document raises its own error, not a warning about how it was read. Metadata read
from a document whose repair moves chunks (a chunk size read as another size) is
marked (see `mark_repaired`), silent or not, so the array stores the repair before it
writes chunks under it. A repair that only respells a value zarr already read the
same way (`true` as 1, `4.0` as 4) moves no chunks, so it is not marked.

To read another kind of invalid document, add a repair to `ARRAY_REPAIRS`.
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
    """How a repair read a stored document."""

    moves_chunks: bool
    """Whether chunks written under the repaired document are not where a reader of the
    stored document looks for them, so the array must store the repair before it
    writes chunks."""
    warning: str | None
    """What the user must act on, if anything."""


type Repair = Callable[[ArrayDocument], tuple[ArrayDocument, Reading] | None]
"""Returns `None` if the document needs no repair, else the repaired document and how
it was read."""

RECREATE_HINT: Final = (
    "The array holds only its fill value, so recreate it with the chunk shape you want; "
    "nothing is lost: `zarr.from_array(array.store, name=array.path, data=array, "
    "chunks=<chunk shape>, overwrite=True, write_data=False)` keeps its data type, fill "
    "value, attributes and codecs. To keep the array as it is instead, store valid metadata "
    "with `array.update_attributes({})`. Either way, open the array writable first, and if a "
    "group holds consolidated metadata for the array, then also call "
    "`zarr.consolidate_metadata` on that group."
)
"""How to act on a document whose array can hold no data: a stored chunk size of 0 is read
as the smallest chunk size, which is a poor chunk shape for the data the array grows into,
and a re-save would keep it."""


def mark_repaired[M](
    metadata: M,
    stored: ArrayDocument,
    readings: Sequence[Reading],
    path: str | None,
    *,
    warn: bool = True,
) -> M:
    """Record that `metadata` was read from the document `stored`, which needed the
    repairs whose `readings` `repair_array_document` returned, if any. If a reading
    moves chunks, keep a copy of `stored` as `_stored_document` on `metadata`, so the
    array stores the repair before it writes chunks under it and consolidated metadata
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


def _read_chunk_size(
    size: JSON, span: int | None, unit: int
) -> tuple[int, bool, str | None] | None:
    """Read one entry of a stored regular chunk shape as a chunk edge length.

    Returns the edge length, whether it moves chunks (it is not the size a reader of
    the stored entry uses), and, where the user must act on how it was read, how it was
    read; `None` if the entry cannot be read, which leaves it for the metadata
    constructors to check. A JSON int >= 1 is kept, JSON `true` is read as 1, and 0 or
    JSON `false` is read as `unit`, the smallest chunk edge length the axis can have (1,
    or the inner chunk size of a shard), whatever the length `span` of the axis. That is
    what `chunks=-1` gives on an axis of length 0, where these sizes were written, and it
    does not depend on how far the axis has grown since. On an axis of positive length no
    chunk can have been stored under a chunk size of 0, so the array holds only its fill
    value. `span` is `None` where no stored 0 is known, as in the inner chunk shape of a
    sharding codec: 0 is then left as stored.
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
    return None


def _read_chunk_shape(
    stored: JSON, spans: Sequence[int | None], units: Iterable[int] = ()
) -> tuple[list[int], Reading] | None:
    """Read a stored regular chunk shape, entry by entry (see `_read_chunk_size`), for
    axes of lengths `spans` whose chunks are multiples of `units` (1 where not given).

    Returns the chunk shape and how it was read, if any entry was read as another value
    (the warning says how the chunk shape was read and, as the array then holds only its
    fill value, recommends recreating it, see `RECREATE_HINT`); `None` if it cannot be
    read or needs no repair.
    """
    if not (isinstance(stored, list) and len(stored) == len(spans)):
        return None
    edges: list[int] = []
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
        return None
    warning = (
        f"The stored chunk shape {json.dumps(stored)} is invalid: chunk sizes must be "
        f"integers of at least 1. It is read as {edges}, reading {'; '.join(readings)}. "
        f"{RECREATE_HINT}"
        if readings
        else None
    )
    return edges, Reading(moves_chunks, warning)


def _invalid_chunk_sizes_v2(doc: ArrayDocument) -> tuple[ArrayDocument, Reading] | None:
    shape = doc.get("shape")
    if not _is_int_list(shape):
        return None
    match _read_chunk_shape(doc.get("chunks"), shape):
        case chunks, reading:
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
            repaired: dict[str, JSON] = {}
            stored = configuration.get("chunk_shape")
            if isinstance(stored, list) and (
                read := _read_chunk_shape(stored, [None] * len(stored))
            ):
                repaired["chunk_shape"] = read[0]
            if isinstance(codecs := configuration.get("codecs"), list) and (
                inner := _read_codecs(codecs)
            ):
                repaired["codecs"] = inner
            if not repaired:
                return None
            # The mapping pattern does not narrow `codec` for mypy.
            return {
                **cast("Mapping[str, JSON]", codec),
                "configuration": {**configuration, **repaired},
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
    match _read_chunk_shape(configuration.get("chunk_shape"), shape, units):
        case chunk_shape, reading:
            repaired = {**configuration, "chunk_shape": chunk_shape}
            return {**doc, "chunk_grid": {**grid, "configuration": repaired}}, reading
    return None


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
    repaired = {**configuration, "chunk_shapes": read}
    # A stored edge is read as the value it equals, so the reading moves no chunks.
    return {**doc, "chunk_grid": {**grid, "configuration": repaired}}, Reading(False, None)


ARRAY_REPAIRS: Final[Mapping[ZarrFormat, tuple[Repair, ...]]] = {
    2: (_invalid_chunk_sizes_v2,),
    # The inner chunk shape is read first: it gives the unit of the outer chunk shape.
    # The rectilinear edge lengths are read last, after any repair that yields a
    # rectilinear chunk grid.
    3: (_invalid_inner_chunk_sizes_v3, _invalid_chunk_sizes_v3, _invalid_edge_lengths_v3),
}
"""The repairs of an array document of each Zarr format, applied in order."""


def repair_array_document(
    doc: ArrayDocument, zarr_format: ZarrFormat
) -> tuple[ArrayDocument, list[Reading]]:
    """Apply the repairs for `zarr_format` to a stored array metadata document.

    Returns the repaired document and the reading of each repair that changed it, for
    `mark_repaired` once the document has been validated.
    """
    readings: list[Reading] = []
    for repair in ARRAY_REPAIRS[zarr_format]:
        repaired = repair(doc)
        if repaired is not None:
            doc, reading = repaired
            readings.append(reading)
    return doc, readings
