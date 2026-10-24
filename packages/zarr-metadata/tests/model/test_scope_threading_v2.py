"""Tests that every v2 entry point reads in the scope it is given, uses `CORE_V2` when given none, and refuses a scope of another format."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

import zarr_metadata.model as zm
from zarr_metadata.model import ZarrV2ArrayMetadata
from zarr_metadata.problems import MetadataValidationError
from zarr_metadata.v2.definition import (
    CORE_V2,
    ZarrV2Context,
)
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr_metadata.definition import Context

EMPTY = ZarrV2Context()
BASE: dict[str, Any] = dict(ZarrV2ArrayMetadata.create_default(shape=(4,), chunks=(2,)).to_json())
BAD: dict[str, Any] = {**BASE, "compressor": {"id": "gzip", "level": "x"}}
"""An array that the v2 scope refuses (a gzip level that is not an integer) and that an empty scope does not check."""
GROUP: dict[str, Any] = {"zarr_format": 2}
CONSOLIDATED: dict[str, Any] = {
    "zarr_consolidated_format": 1,
    "metadata": {".zgroup": GROUP, "a/.zarray": BAD},
}
STORE_A = {".zarray": json.dumps(BAD).encode()}
STORE_G = {".zgroup": json.dumps(GROUP).encode()}
STORE_C = {".zmetadata": json.dumps(CONSOLIDATED).encode()}


def _accepts(read: Callable[..., object]) -> Callable[[Context | None], bool]:
    """Return whether `read`, given `context`, finds nothing wrong. Success is a model, a reading with a model, or a field that was read or left unclaimed."""

    def accepted(context: Context | None) -> bool:
        try:
            found = read(context=context)
        except MetadataValidationError:
            return False
        if isinstance(found, tuple) and len(found) == 2 and isinstance(found[1], tuple):
            return len(found[1]) == 0  # a resolved field and its problems
        reading = getattr(found, "reading", found)
        metadata = getattr(reading, "metadata", found)
        return metadata is not None

    return accepted


ENTRY_POINTS: dict[str, Callable[..., object]] = {
    "read_array_metadata_v2": lambda context: zm.read_array_metadata_v2(BAD, context=context),
    "ZarrV2ArrayMetadata": lambda context: zm.ZarrV2ArrayMetadata(BAD, context=context),
    "ZarrV2ArrayMetadata.from_json": lambda context: zm.ZarrV2ArrayMetadata.from_json(
        BAD, context=context
    ),
    "ZarrV2ArrayMetadata.from_key_value": lambda context: zm.ZarrV2ArrayMetadata.from_key_value(
        STORE_A, context=context
    ),
    "ZarrV2ArrayMetadata.create_default": lambda context: zm.ZarrV2ArrayMetadata.create_default(
        shape=(4,), chunks=(2,), compressor=BAD["compressor"], context=context
    ),
    "ZarrV2ArrayMetadata.with_context": lambda context: zm.ZarrV2ArrayMetadata(
        BAD, context=EMPTY
    ).with_context(context),
    "ZarrV2ArrayMetadata.refined_in": lambda context: zm.ZarrV2ArrayMetadata(
        BAD, context=EMPTY
    ).refined_in(context),
    "ZarrV2GroupMetadata": lambda context: zm.ZarrV2GroupMetadata(GROUP, context=context),
    "ZarrV2GroupMetadata.from_json": lambda context: zm.ZarrV2GroupMetadata.from_json(
        GROUP, context=context
    ),
    "ZarrV2GroupMetadata.from_key_value": lambda context: zm.ZarrV2GroupMetadata.from_key_value(
        STORE_G, context=context
    ),
    "ZarrV2GroupMetadata.create_default": lambda context: zm.ZarrV2GroupMetadata.create_default(
        context=context
    ),
    "ZarrV2ConsolidatedMetadata": lambda context: zm.ZarrV2ConsolidatedMetadata(
        CONSOLIDATED, context=context
    ),
    "ZarrV2ConsolidatedMetadata.from_json": lambda context: zm.ZarrV2ConsolidatedMetadata.from_json(
        CONSOLIDATED, context=context
    ),
    "ZarrV2ConsolidatedMetadata.from_key_value": (
        lambda context: zm.ZarrV2ConsolidatedMetadata.from_key_value(STORE_C, context=context)
    ),
    "read_repaired_consolidated_metadata_v2": (
        lambda context: zm.read_repaired_consolidated_metadata_v2(CONSOLIDATED, context=context)
    ),
    "read_repaired_node_metadata_v2": lambda context: zm.read_repaired_node_metadata_v2(
        {".zarray": BAD}, context=context
    ),
    "read_group_metadata_v2": lambda context: zm.read_group_metadata_v2(GROUP, context=context),
    "read_consolidated_metadata_v2": lambda context: zm.read_consolidated_metadata_v2(
        CONSOLIDATED, context=context
    ),
    "read_node_metadata_v2": lambda context: zm.read_node_metadata_v2(
        {".zarray": BAD}, context=context
    ),
    "node_metadata_from_key_value_v2": lambda context: zm.node_metadata_from_key_value_v2(
        STORE_A, context=context
    ),
}
"""Each v2 entry point, reading a document that contains the refused compressor wherever that document type allows one."""

REFUSES_BAD = frozenset(
    name for name in ENTRY_POINTS if "Group" not in name and "group" not in name
)
"""The entry points whose document contains the refused compressor. The other entry points accept their document in every scope."""


@pytest.mark.parametrize("name", ENTRY_POINTS.keys())
def test_every_v2_entry_point_reads_in_the_scope_it_is_given(name: str) -> None:
    """Each v2 entry point accepts its document in an empty scope. In `CORE_V2` (also selected by `None`), it refuses the non-integer gzip level when the document contains it. This shows that each entry point reads in the scope it is given."""
    accepted = _accepts(ENTRY_POINTS[name])
    assert accepted(EMPTY) is True
    assert accepted(CORE_V2) is (name not in REFUSES_BAD)
    assert accepted(None) is (name not in REFUSES_BAD)


@pytest.mark.parametrize("name", ENTRY_POINTS.keys())
def test_error_every_v2_entry_point_refuses_a_scope_of_another_format(name: str) -> None:
    """A v2 entry point given a v3 scope raises `TypeError`. A scope reads documents of one format, and a v3 scope claims nothing that a v2 document contains."""
    with pytest.raises(TypeError, match="format"):
        ENTRY_POINTS[name](context=CORE_AND_EXTENSIONS)


NOT_JSON = b"{not json"
STORE_READERS: dict[str, Callable[..., object]] = {
    "ZarrV2ArrayMetadata.from_key_value": lambda context: zm.ZarrV2ArrayMetadata.from_key_value(
        {".zarray": NOT_JSON}, context=context
    ),
    "ZarrV2GroupMetadata.from_key_value": lambda context: zm.ZarrV2GroupMetadata.from_key_value(
        {".zgroup": NOT_JSON}, context=context
    ),
    "ZarrV2ConsolidatedMetadata.from_key_value": (
        lambda context: zm.ZarrV2ConsolidatedMetadata.from_key_value(
            {".zmetadata": NOT_JSON}, context=context
        )
    ),
    "node_metadata_from_key_value_v2": lambda context: zm.node_metadata_from_key_value_v2(
        {".zarray": NOT_JSON}, context=context
    ),
}
"""Each v2 entry point that decodes a store's bytes, given bytes that are not JSON."""


@pytest.mark.parametrize("name", STORE_READERS.keys())
def test_error_a_v2_store_reader_refuses_a_scope_of_another_format_before_decoding(
    name: str,
) -> None:
    """A v2 entry point that reads a store's bytes raises `TypeError` for a v3 scope before it decodes the bytes, so the wrong scope is reported even when the bytes are not JSON. With a v2 scope, the same bytes raise `MetadataValidationError`."""
    with pytest.raises(TypeError, match="format"):
        STORE_READERS[name](context=CORE_AND_EXTENSIONS)
    with pytest.raises(MetadataValidationError):
        STORE_READERS[name](context=CORE_V2)
