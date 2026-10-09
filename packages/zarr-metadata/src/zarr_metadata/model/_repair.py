"""Repairing v3 metadata that a known writer bug made invalid.

The readers are strict: a document the spec does not allow has problems,
whoever wrote it. Some writers have written documents the spec does not
allow, in ways that say plainly what they meant. Repairing one is a step
of its own, before a strict read, which a caller asks for by name:
`read_repaired_node_metadata_v3` rather than `read_node_metadata_v3`.

What is repaired is a closed set, each member of it a writer's bug: the
shape of the JSON it wrote is a TypedDict, which a document must have for
the repair to apply, and the repair makes it correct JSON. Anything else
-- a missing `data_type`, a chunk length of 0 along a dimension that is
not empty -- is left as it is, for the strict read to report.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias, TypedDict, cast

from annotated_types import Ge

from zarr_metadata._common import (
    JSONValue,
)
from zarr_metadata._json import (
    JSON_DEPTH,
    MetadataValidationError,
    ValidationProblem,
    is_json_object,
    is_object,
)
from zarr_metadata._typed_json import Loc, check
from zarr_metadata.model._group import (
    ZarrV2ConsolidatedMetadata,
    ZarrV3NodeMetadataReading,
    read_node_metadata_v3,
)
from zarr_metadata.v2.definition import CORE_V2
from zarr_metadata.v2.group import ZARR_V2_GROUP_METADATA_STORE_KEY
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, ZarrV2Context, ZarrV3Context, scoped
from zarr_metadata.v3.consolidated import ZARR_V3_CONSOLIDATED_METADATA_KEY

RepairKind: TypeAlias = Literal[
    "zero_chunk_length", "null_consolidated_metadata", "consolidated_metadata_in_zgroup_entry"
]
"""Each writer bug a repair undoes, by name."""


@dataclass(frozen=True, slots=True)
class Repair:
    """One change a repair made to a document: where, which bug it undid, and what it did."""

    loc: Loc
    """Where the change is, in the document: the value changed, or the key removed."""
    kind: RepairKind
    """The writer bug the change undoes."""
    message: str
    """What was written, by which writer, and what it became."""


_Length = Annotated[int, Ge(0)]


class ZarrV3ZeroChunkRegularGridConfigurationJSON(TypedDict):
    """A `regular` grid's configuration as zarr-python 3.0 and 3.1 wrote it: chunk lengths that may be 0."""

    chunk_shape: tuple[_Length, ...]


class ZarrV3ZeroChunkRegularGridJSON(TypedDict):
    """A `regular` chunk grid whose chunk lengths may be 0."""

    name: Literal["regular"]
    configuration: ZarrV3ZeroChunkRegularGridConfigurationJSON


class ZarrV3ZeroChunkArrayMetadataJSON(TypedDict):
    """The members of an array document the `zero_chunk_length` repair reads.

    zarr-python 3.0 and 3.1 wrote a chunk length of 0 along a dimension of
    length 0, which the regular grid refuses: "Chunk sizes must be greater
    than zero". A chunk along an empty dimension holds nothing whatever its
    length, so the repair writes 1 there, as zarr-python has since
    (https://github.com/zarr-developers/zarr-python/pull/4328).
    """

    node_type: Literal["array"]
    shape: tuple[_Length, ...]
    chunk_grid: ZarrV3ZeroChunkRegularGridJSON


class ZarrV3NullConsolidatedGroupMetadataJSON(TypedDict):
    """The members of a group document the `null_consolidated_metadata` repair reads.

    zarr-python 3.0 and 3.1 wrote `"consolidated_metadata": null` on a group
    they had not consolidated, which the convention does not allow: the member is
    an object, or absent. The repair removes it, which is what the writer
    meant.
    """

    node_type: Literal["group"]
    consolidated_metadata: None


def repair_node_metadata_v3(value: object) -> tuple[object, tuple[Repair, ...]]:
    """`value`, a v3 `zarr.json`, with each known writer bug in it undone, and what was changed.

    Each document consolidated metadata holds is repaired too. What no
    repair applies to is left as it is, and `value` is not changed: a
    repaired document is a new one, sharing what it did not change with
    `value`. Repairing a document with none of the bugs gives it back,
    and no repairs.
    """
    return _repaired(value, ())


def _repaired(value: object, at: Loc) -> tuple[object, tuple[Repair, ...]]:
    if not is_json_object(value) or len(at) >= JSON_DEPTH:
        # Not a document, or past the levels a reader walks, which the
        # strict read reports.
        return value, ()
    original = value
    repairs: list[Repair] = []
    document = _zero_chunk_length(original, at, repairs)
    document = _null_consolidated_metadata(document, at, repairs)
    document = _consolidated(document, at, repairs)
    return (original if len(repairs) == 0 else dict(document)), tuple(repairs)


def _members(document: Mapping[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    """The members of `document` a repair reads: only those, so the rest is not walked."""
    return {key: document[key] for key in keys if key in document}


def _zero_chunk_length(
    document: Mapping[str, object], at: Loc, repairs: list[Repair]
) -> Mapping[str, object]:
    array, problems = check(
        _members(document, ("node_type", "shape", "chunk_grid")), ZarrV3ZeroChunkArrayMetadataJSON
    )
    if array is None or len(problems) != 0:
        return document
    shape = array["shape"]
    chunk_shape = array["chunk_grid"]["configuration"]["chunk_shape"]
    zeros = [axis for axis, length in enumerate(chunk_shape) if length == 0]
    if len(zeros) == 0 or len(chunk_shape) != len(shape) or any(shape[a] != 0 for a in zeros):
        # Nothing to repair, or a 0 that is no writer's bug: the strict
        # read reports it.
        return document
    repairs.extend(
        Repair(
            (*at, "chunk_grid", "configuration", "chunk_shape", axis),
            "zero_chunk_length",
            "a chunk length of 0 along a dimension of length 0, as zarr-python 3.0 and 3.1 "
            "wrote it, written as 1",
        )
        for axis in zeros
    )
    grid = cast("Mapping[str, object]", document["chunk_grid"])
    configuration = cast("Mapping[str, object]", grid["configuration"])
    repaired = tuple(max(length, 1) for length in chunk_shape)
    return {
        **document,
        "chunk_grid": {**grid, "configuration": {**configuration, "chunk_shape": repaired}},
    }


def _null_consolidated_metadata(
    document: Mapping[str, object], at: Loc, repairs: list[Repair]
) -> Mapping[str, object]:
    group, problems = check(
        _members(document, ("node_type", ZARR_V3_CONSOLIDATED_METADATA_KEY)),
        ZarrV3NullConsolidatedGroupMetadataJSON,
    )
    if group is None or len(problems) != 0:
        return document
    repairs.append(
        Repair(
            (*at, ZARR_V3_CONSOLIDATED_METADATA_KEY),
            "null_consolidated_metadata",
            "a consolidated_metadata of null, as zarr-python 3.0 and 3.1 wrote it, removed",
        )
    )
    return {key: item for key, item in document.items() if key != ZARR_V3_CONSOLIDATED_METADATA_KEY}


def _consolidated(
    document: Mapping[str, object], at: Loc, repairs: list[Repair]
) -> Mapping[str, object]:
    """`document` with each document its consolidated metadata holds repaired, where it holds any."""
    member = document.get(ZARR_V3_CONSOLIDATED_METADATA_KEY)
    if not is_object(member):
        return document
    envelope = member
    entries = envelope.get("metadata")
    if not is_object(entries):
        return document
    held: dict[object, object] = {}
    found = len(repairs)
    for key, entry in entries.items():
        if isinstance(key, str):
            entry, inside = _repaired(
                entry, (*at, ZARR_V3_CONSOLIDATED_METADATA_KEY, "metadata", key)
            )
            repairs.extend(inside)
        held[key] = entry
    if len(repairs) == found:
        return document
    return {**document, ZARR_V3_CONSOLIDATED_METADATA_KEY: {**envelope, "metadata": held}}


@dataclass(frozen=True, slots=True)
class ZarrV3RepairedNodeMetadataReading:
    """A v3 `zarr.json` read after its known writer bugs were undone: the strict reading of the repaired document, and the repairs."""

    reading: ZarrV3NodeMetadataReading
    """The repaired document, as `read_node_metadata_v3` reads it: its problems are the repaired document's, and its model when there are none."""
    repairs: tuple[Repair, ...]
    """What was changed to make the document that was read."""


def read_repaired_node_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> ZarrV3RepairedNodeMetadataReading:
    """`value`, a v3 `zarr.json`, read in `context` as `read_node_metadata_v3` reads it, once `repair_node_metadata_v3` has undone each known writer bug in it.

    For a reader of stores other writers made, which asks for repairs by
    calling this rather than `read_node_metadata_v3`. Whatever no repair
    applies to is read as it is, and reported as `read_node_metadata_v3`
    reports it.
    """
    scope = scoped(context, CORE_AND_EXTENSIONS)
    repaired, repairs = repair_node_metadata_v3(value)
    return ZarrV3RepairedNodeMetadataReading(
        read_node_metadata_v3(repaired, context=scope), repairs
    )


class ZarrV2ZGroupWithConsolidatedMetadataJSON(TypedDict):
    """A `.zgroup` entry of a `.zmetadata` as zarr-python 3.x writes one below the root: with a `consolidated_metadata` member, which a v2 group document does not take."""

    zarr_format: Literal[2]
    consolidated_metadata: Mapping[str, JSONValue]


def repair_consolidated_metadata_v2(value: object) -> tuple[object, tuple[Repair, ...]]:
    """`value`, a v2 `.zmetadata`, with each known writer bug in it undone, and what was changed.

    zarr-python 3.x writes a `consolidated_metadata` member into each
    `.zgroup` entry below the root, which is removed; the root's is left,
    since no writer puts one there. What no repair
    applies to is left as it is, and `value` is not changed; a document
    with none of the bugs is given back, and no repairs.
    """
    if not is_json_object(value):
        return value, ()
    document = value
    entries = document.get("metadata")
    if not is_object(entries):
        return value, ()
    repairs: list[Repair] = []
    held: dict[object, object] = {}
    for key, entry in entries.items():
        if isinstance(key, str):
            path, _, name = key.rpartition("/")
            # Below the root only: no writer puts the member in the root's .zgroup.
            if name == ZARR_V2_GROUP_METADATA_STORE_KEY and path.strip("/") != "":
                entry = _without_consolidated_metadata(entry, ("metadata", key), repairs)
        held[key] = entry
    if len(repairs) == 0:
        return value, ()
    return {**document, "metadata": held}, tuple(repairs)


def _without_consolidated_metadata(entry: object, at: Loc, repairs: list[Repair]) -> object:
    """`entry`, a `.zgroup` entry, without the member zarr-python 3.x writes into it, when it is one such."""
    if not is_json_object(entry):
        return entry
    group = entry
    shaped, problems = check(
        _members(group, ("zarr_format", ZARR_V3_CONSOLIDATED_METADATA_KEY)),
        ZarrV2ZGroupWithConsolidatedMetadataJSON,
    )
    if shaped is None or len(problems) != 0:
        return entry
    repairs.append(
        Repair(
            (*at, ZARR_V3_CONSOLIDATED_METADATA_KEY),
            "consolidated_metadata_in_zgroup_entry",
            "a consolidated_metadata, as zarr-python 3.x writes into a .zgroup entry of a "
            ".zmetadata, removed",
        )
    )
    kept: dict[str, object] = {
        key: item for key, item in group.items() if key != ZARR_V3_CONSOLIDATED_METADATA_KEY
    }
    return kept


@dataclass(frozen=True, slots=True)
class ZarrV2RepairedConsolidatedMetadataReading:
    """A v2 `.zmetadata` read after its known writer bugs were undone: the repaired document's problems, its model when there are none, and the repairs."""

    problems: tuple[ValidationProblem, ...]
    """Every problem of the repaired document."""
    metadata: ZarrV2ConsolidatedMetadata | None
    """The repaired document's model, when it has no problem; None otherwise."""
    repairs: tuple[Repair, ...]
    """What was changed to make the document that was read."""


def read_repaired_consolidated_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> ZarrV2RepairedConsolidatedMetadataReading:
    """`value`, a v2 `.zmetadata`, read in `context` as `ZarrV2ConsolidatedMetadata` reads it, once `repair_consolidated_metadata_v2` has undone each known writer bug in it.

    For a reader of stores other writers made, which asks for repairs by
    calling this rather than the strict model. Whatever no repair applies
    to is read as it is, and reported as the strict read reports it.
    """
    scope = scoped(context, CORE_V2)
    repaired, repairs = repair_consolidated_metadata_v2(value)
    try:
        model: ZarrV2ConsolidatedMetadata | None = ZarrV2ConsolidatedMetadata(repaired, scope)
    except MetadataValidationError as error:
        return ZarrV2RepairedConsolidatedMetadataReading(error.problems, None, repairs)
    return ZarrV2RepairedConsolidatedMetadataReading((), model, repairs)


__all__ = [
    "Repair",
    "RepairKind",
    "ZarrV2RepairedConsolidatedMetadataReading",
    "ZarrV2ZGroupWithConsolidatedMetadataJSON",
    "ZarrV3NullConsolidatedGroupMetadataJSON",
    "ZarrV3RepairedNodeMetadataReading",
    "ZarrV3ZeroChunkArrayMetadataJSON",
    "ZarrV3ZeroChunkRegularGridConfigurationJSON",
    "ZarrV3ZeroChunkRegularGridJSON",
    "read_repaired_consolidated_metadata_v2",
    "read_repaired_node_metadata_v3",
    "repair_consolidated_metadata_v2",
    "repair_node_metadata_v3",
]
