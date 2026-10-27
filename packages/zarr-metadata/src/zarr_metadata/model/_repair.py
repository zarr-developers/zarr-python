"""Repairing v2 and v3 metadata that a known writer bug made invalid.

The readers are strict: a document the spec does not allow has problems,
no matter who wrote it. Some writers have written documents the spec does
not allow, but where the intended meaning is clear. Repair is a separate
step that runs before a strict read, and a caller must request it by
name: `read_repaired_node_metadata_v3` instead of `read_node_metadata_v3`,
or `read_repaired_node_metadata_v2` instead of `read_node_metadata_v2`.

The set of repairs is closed, and each repair targets one writer bug. A
TypedDict describes the shape of the JSON the buggy writer produced. A
document must match that TypedDict for the repair to apply, and the
repair turns it into correct JSON. Anything else (a missing `data_type`,
a chunk length of 0 along a dimension that is not empty) is left as it
is, and the strict read reports it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias, TypedDict, cast

from annotated_types import Ge

from zarr_metadata._common import (
    JSONValue,
)
from zarr_metadata._json import (
    JSON_DEPTH,
    is_json_object,
    is_list_or_tuple,
    is_object,
)
from zarr_metadata._typed_json import Loc, check
from zarr_metadata.definition._context import (
    scoped,
)
from zarr_metadata.model._group_v2 import (
    ZarrV2ConsolidatedMetadataReading,
    ZarrV2NodeMetadataReading,
    read_consolidated_metadata_v2,
    read_node_metadata_v2,
)
from zarr_metadata.model._group_v3 import ZarrV3NodeMetadataReading, read_node_metadata_v3
from zarr_metadata.v2.array import ZARR_V2_ARRAY_METADATA_STORE_KEY
from zarr_metadata.v2.definition import CORE_V2
from zarr_metadata.v2.group import ZARR_V2_GROUP_METADATA_STORE_KEY
from zarr_metadata.v3._core import CORE_AND_EXTENSIONS
from zarr_metadata.v3.consolidated import ZARR_V3_CONSOLIDATED_METADATA_KEY

if TYPE_CHECKING:
    from zarr_metadata.v2._context import ZarrV2Context
    from zarr_metadata.v3._context import ZarrV3Context

RepairKind: TypeAlias = Literal[
    "zero_chunk_length",
    "null_consolidated_metadata",
    "consolidated_metadata_in_zgroup_entry",
    "non_finite_fill_value",
]
"""The name of each writer bug that a repair undoes."""


@dataclass(frozen=True, slots=True)
class Repair:
    """One change a repair made to a document: its location, the bug it undid, and what it changed."""

    loc: Loc
    """The location of the change in the document: the value that changed, or the key that was removed."""
    kind: RepairKind
    """The writer bug the change undoes."""
    message: str
    """What was written, which writer wrote it, and what it was changed to."""


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
    than zero". A chunk along an empty dimension holds nothing, whatever
    its length, so the repair writes 1 there. zarr-python has also written
    1 since https://github.com/zarr-developers/zarr-python/pull/4328.
    """

    node_type: Literal["array"]
    shape: tuple[_Length, ...]
    chunk_grid: ZarrV3ZeroChunkRegularGridJSON


class ZarrV3NullConsolidatedGroupMetadataJSON(TypedDict):
    """The members of a group document the `null_consolidated_metadata` repair reads.

    zarr-python 3.0 and 3.1 wrote `"consolidated_metadata": null` on a group
    they had not consolidated. The convention does not allow this: the member
    is an object, or absent. The repair removes the member, which is what the
    writer meant.
    """

    node_type: Literal["group"]
    consolidated_metadata: None


class ZarrV3NonFiniteFillValueArrayMetadataJSON(TypedDict):
    """The members of an array document the v3 `non_finite_fill_value` repair reads.

    zarr-python 3.0.0 to 3.0.6 wrote the fill value of a float64 array, in
    the documents a group's consolidated metadata lists, as a bare `NaN`,
    `Infinity` or `-Infinity`, which is not JSON. The spec writes these as
    the strings `"NaN"`, `"Infinity"` and `"-Infinity"`. The repair writes
    each such fill value, or complex fill value component, of a float or
    complex data type by its name.
    """

    node_type: Literal["array"]
    data_type: Literal["float16", "float32", "float64", "complex64", "complex128"]


def repair_node_metadata_v3(value: object) -> tuple[object, tuple[Repair, ...]]:
    """Undo each known writer bug in `value`, a v3 `zarr.json`, and return the result with the list of changes.

    Each document in the consolidated metadata is repaired too. A group
    does not read those documents, but `ZarrV3ConsolidatedMetadata` reads
    them from the repaired group's member. Content that no repair applies
    to is left as it is. `value` is not modified: a repaired document is a
    new object, which shares its unchanged parts with `value`. A document
    with none of the bugs is returned as it is, with no repairs.
    """
    return _repaired(value, ())


def _repaired(value: object, at: Loc) -> tuple[object, tuple[Repair, ...]]:
    if not is_json_object(value) or len(at) >= JSON_DEPTH:
        # Not a document, or nested deeper than a reader walks. The strict
        # read reports both.
        return value, ()
    original = value
    repairs: list[Repair] = []
    document = _zero_chunk_length(original, at, repairs)
    if len(at) != 0:
        # The writer bug is in the documents a listing holds only. Those
        # versions wrote a node's own `zarr.json` correctly.
        document = _non_finite_fill_value_v3(document, at, repairs)
    document = _null_consolidated_metadata(document, at, repairs)
    document = _consolidated(document, at, repairs)
    return (original if len(repairs) == 0 else dict(document)), tuple(repairs)


def _non_finite_fill_value_v3(
    document: Mapping[str, object], at: Loc, repairs: list[Repair]
) -> Mapping[str, object]:
    """`document` with a non-finite fill value of a float data type, or complex fill value component, written by its name."""
    array, problems = check(
        _members(document, ("node_type", "data_type")), ZarrV3NonFiniteFillValueArrayMetadataJSON
    )
    if array is None or len(problems) != 0 or "fill_value" not in document:
        return document
    return _non_finite_named(document, array["data_type"].startswith("float"), at, repairs)


def _members(document: Mapping[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    """Only the members of `document` that a repair reads, so the rest is not walked."""
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
        # Nothing to repair, or a 0 that is not a known writer bug. The
        # strict read reports the latter.
        return document
    repairs.extend(
        Repair(
            (*at, "chunk_grid", "configuration", "chunk_shape", axis),
            "zero_chunk_length",
            "a regular grid's chunk length of 0 along a dimension of length 0, as zarr-python "
            "3.0 and 3.1 wrote it, written as 1",
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
    """`document` with each document in its consolidated metadata repaired, if it has any."""
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
    """The repaired document, as `read_node_metadata_v3` reads it. The problems are those of the repaired document. The model is present when there are no problems."""
    repairs: tuple[Repair, ...]
    """The changes that produced the document that was read."""


def read_repaired_node_metadata_v3(
    value: object, *, context: ZarrV3Context | None = None
) -> ZarrV3RepairedNodeMetadataReading:
    """Read `value`, a v3 `zarr.json`, in `context` as `read_node_metadata_v3` does, after `repair_node_metadata_v3` has undone each known writer bug in it.

    Use this to read stores that other writers made. A caller requests
    repairs by calling this function instead of `read_node_metadata_v3`.
    Content that no repair applies to is read as it is, and reported the
    same way `read_node_metadata_v3` reports it.
    """
    context = scoped(context, CORE_AND_EXTENSIONS)
    repaired, repairs = repair_node_metadata_v3(value)
    return ZarrV3RepairedNodeMetadataReading(
        read_node_metadata_v3(repaired, context=context), repairs
    )


class ZarrV2ZGroupWithConsolidatedMetadataJSON(TypedDict):
    """A `.zgroup` entry of a `.zmetadata`, below the root, as zarr-python 3.x writes it: with a `consolidated_metadata` member, which a v2 group document does not allow."""

    zarr_format: Literal[2]
    consolidated_metadata: Mapping[str, JSONValue]


def repair_consolidated_metadata_v2(value: object) -> tuple[object, tuple[Repair, ...]]:
    """Undo each known writer bug in `value`, a v2 `.zmetadata`, and return the result with the list of changes.

    zarr-python 3.x writes a `consolidated_metadata` member into each
    `.zgroup` entry below the root. This function removes it. The root's
    entry is left alone, since no writer puts the member there. Each
    `.zarray` entry is repaired as `repair_node_metadata_v2` repairs a
    `.zarray`. Content that no repair applies to is left as it is, and
    `value` is not modified. A document with none of the bugs is returned
    as it is, with no repairs.
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
            if name == ZARR_V2_ARRAY_METADATA_STORE_KEY:
                entry = _repaired_zarray(entry, ("metadata", key), repairs)
        held[key] = entry
    if len(repairs) == 0:
        return value, ()
    return {**document, "metadata": held}, tuple(repairs)


def _without_consolidated_metadata(entry: object, at: Loc, repairs: list[Repair]) -> object:
    """`entry`, a `.zgroup` entry, without the member zarr-python 3.x writes into it, if the entry has that member."""
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


_ZarrV2FloatTypestr: TypeAlias = Literal[
    "<f2", ">f2", "<f4", ">f4", "<f8", ">f8", "<c8", ">c8", "<c16", ">c16"
]


class ZarrV2NonFiniteFillValueArrayMetadataJSON(TypedDict):
    """The members of a `.zarray` that the `non_finite_fill_value` repair reads, other than the fill value: a float or complex `dtype`.

    zarr-python 3.0.0 to 3.0.6 wrote a non-finite fill value of a float
    array as a bare token, `"fill_value": NaN`, and each non-finite
    component of a complex one the same way, `[NaN, Infinity]`. Neither
    is JSON. The v2 spec spells these values as strings, `"NaN"`,
    `"Infinity"` and `"-Infinity"`
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L178-L190),
    and so does zarr-python since 3.0.7. Python's `json` module reads the
    tokens as floats, which `check` refuses, so the fill value is not a
    member here: the repair reads a float, or a pair for a complex dtype.
    """

    zarr_format: Literal[2]
    dtype: _ZarrV2FloatTypestr


def repair_node_metadata_v2(
    files: Mapping[str, object],
) -> tuple[Mapping[str, object], tuple[Repair, ...]]:
    """Undo each known writer bug in `files`, a v2 node's files as `read_node_metadata_v2` takes them, and return the result with the list of changes.

    zarr-python 3.0.0 to 3.0.6 wrote a non-finite fill value in a
    `.zarray` as a bare token, `NaN`, which Python's `json` module reads as
    a float. This function writes it as the string the v2 spec uses,
    `"NaN"`, and does the same for each component of a complex fill value.
    Each repair is located in `files`, under the name of the file. Content
    that no repair applies to is left as it is, and `files` is not
    modified. Files with none of the bugs are returned as they are, with
    no repairs.
    """
    if ZARR_V2_ARRAY_METADATA_STORE_KEY not in files:
        return files, ()
    repairs: list[Repair] = []
    zarray = _repaired_zarray(
        files[ZARR_V2_ARRAY_METADATA_STORE_KEY], (ZARR_V2_ARRAY_METADATA_STORE_KEY,), repairs
    )
    if len(repairs) == 0:
        return files, ()
    return {**files, ZARR_V2_ARRAY_METADATA_STORE_KEY: zarray}, tuple(repairs)


def _repaired_zarray(entry: object, at: Loc, repairs: list[Repair]) -> object:
    """`entry`, a `.zarray` document, with each known writer bug in it undone."""
    if not is_json_object(entry):
        return entry
    return _non_finite_fill_value(entry, at, repairs)


def _non_finite_fill_value(
    document: Mapping[str, object], at: Loc, repairs: list[Repair]
) -> Mapping[str, object]:
    """`document` with a non-finite float fill value, or complex fill value component, written by its name."""
    array, problems = check(
        _members(document, ("zarr_format", "dtype")), ZarrV2NonFiniteFillValueArrayMetadataJSON
    )
    if array is None or len(problems) != 0 or "fill_value" not in document:
        return document
    return _non_finite_named(document, array["dtype"][1] == "f", at, repairs)


def _non_finite_named(
    document: Mapping[str, object], is_float: bool, at: Loc, repairs: list[Repair]
) -> Mapping[str, object]:
    """`document`, an array of either format with a `fill_value` of a float data type when `is_float` and of a complex one otherwise, with a non-finite fill value, or complex fill value component, written by its name. Each name written is added to `repairs`, located under `at`."""
    fill_value = document["fill_value"]
    if is_float:
        name = _non_finite_name(fill_value)
        if name is None:
            return document
        repairs.append(_non_finite_repair((*at, "fill_value"), name, "a fill value"))
        return {**document, "fill_value": name}
    if not is_list_or_tuple(fill_value) or len(fill_value) != 2:
        return document
    names = [_non_finite_name(component) for component in fill_value]
    if all(name is None for name in names):
        return document
    repairs.extend(
        _non_finite_repair((*at, "fill_value", index), name, "a fill value component")
        for index, name in enumerate(names)
        if name is not None
    )
    repaired = tuple(
        component if name is None else name
        for component, name in zip(fill_value, names, strict=True)
    )
    return {**document, "fill_value": repaired}


def _non_finite_name(value: object) -> str | None:
    """The name the v2 spec gives `value`, a non-finite float: `"NaN"`, `"Infinity"` or `"-Infinity"`. None for any other value."""
    if not isinstance(value, float) or math.isfinite(value):
        return None
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _non_finite_repair(loc: Loc, name: str, what: str) -> Repair:
    return Repair(
        loc,
        "non_finite_fill_value",
        f'{what} of {name} as a bare token, as zarr-python 3.0.0 to 3.0.6 wrote it, written as "{name}"',
    )


@dataclass(frozen=True, slots=True)
class ZarrV2RepairedNodeMetadataReading:
    """A v2 node's files read after their known writer bugs were undone: the strict reading of the repaired files, and the repairs."""

    reading: ZarrV2NodeMetadataReading
    """The repaired files, as `read_node_metadata_v2` reads them. The problems are those of the repaired files. The model is present when there are no problems."""
    repairs: tuple[Repair, ...]
    """The changes that produced the files that were read."""


def read_repaired_node_metadata_v2(
    files: Mapping[str, object], *, context: ZarrV2Context | None = None
) -> ZarrV2RepairedNodeMetadataReading:
    """Read `files`, a v2 node's files, in `context` as `read_node_metadata_v2` does, after `repair_node_metadata_v2` has undone each known writer bug in them.

    Use this to read stores that other writers made. A caller requests
    repairs by calling this function instead of `read_node_metadata_v2`.
    Content that no repair applies to is read as it is, and reported the
    same way `read_node_metadata_v2` reports it.
    """
    context = scoped(context, CORE_V2)
    repaired, repairs = repair_node_metadata_v2(files)
    return ZarrV2RepairedNodeMetadataReading(
        read_node_metadata_v2(repaired, context=context), repairs
    )


@dataclass(frozen=True, slots=True)
class ZarrV2RepairedConsolidatedMetadataReading:
    """A v2 `.zmetadata` read after its known writer bugs were undone: the strict reading of the repaired document, and the repairs."""

    reading: ZarrV2ConsolidatedMetadataReading
    """The repaired document, as `read_consolidated_metadata_v2` reads it. The problems are those of the repaired document. The model is present when there are no problems."""
    repairs: tuple[Repair, ...]
    """The changes that produced the document that was read."""


def read_repaired_consolidated_metadata_v2(
    value: object, *, context: ZarrV2Context | None = None
) -> ZarrV2RepairedConsolidatedMetadataReading:
    """Read `value`, a v2 `.zmetadata`, in `context` as `read_consolidated_metadata_v2` does, after `repair_consolidated_metadata_v2` has undone each known writer bug in it.

    Use this to read stores that other writers made. A caller requests
    repairs by calling this function instead of
    `read_consolidated_metadata_v2`. Content that no repair applies to is
    read as it is, and reported the same way
    `read_consolidated_metadata_v2` reports it.
    """
    context = scoped(context, CORE_V2)
    repaired, repairs = repair_consolidated_metadata_v2(value)
    return ZarrV2RepairedConsolidatedMetadataReading(
        read_consolidated_metadata_v2(repaired, context=context), repairs
    )


__all__ = [
    "Repair",
    "RepairKind",
    "ZarrV2NonFiniteFillValueArrayMetadataJSON",
    "ZarrV2RepairedConsolidatedMetadataReading",
    "ZarrV2RepairedNodeMetadataReading",
    "ZarrV2ZGroupWithConsolidatedMetadataJSON",
    "ZarrV3NonFiniteFillValueArrayMetadataJSON",
    "ZarrV3NullConsolidatedGroupMetadataJSON",
    "ZarrV3RepairedNodeMetadataReading",
    "ZarrV3ZeroChunkArrayMetadataJSON",
    "ZarrV3ZeroChunkRegularGridConfigurationJSON",
    "ZarrV3ZeroChunkRegularGridJSON",
    "read_repaired_consolidated_metadata_v2",
    "read_repaired_node_metadata_v2",
    "read_repaired_node_metadata_v3",
    "repair_consolidated_metadata_v2",
    "repair_node_metadata_v2",
    "repair_node_metadata_v3",
]
