"""Tests that every entry point reads in the scope it is given, and in `CORE_AND_EXTENSIONS_CONTEXT` when given none."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

import zarr_metadata.model as zm
from zarr_metadata.json_schema import node_metadata_json_schema_v3
from zarr_metadata.model import (
    ZarrV3ArrayMetadata,
)
from zarr_metadata.problems import MetadataValidationError
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS_CONTEXT, ZarrV3Context

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr_metadata.definition import Context

EMPTY = ZarrV3Context()
BASE: dict[str, Any] = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
BAD: dict[str, Any] = {
    **BASE,
    "codecs": (*BASE["codecs"], {"name": "gzip", "configuration": {"level": 12}}),
}
"""An array that the default scope refuses (a gzip level of 12) and that an empty scope does not check."""
GROUP: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "group",
    "consolidated_metadata": {"kind": "inline", "must_understand": False, "metadata": {"a": BAD}},
}
STORE_A = {"zarr.json": json.dumps(BAD).encode()}
STORE_G = {"zarr.json": json.dumps(GROUP).encode()}


def _accepts(read: Callable[..., object]) -> Callable[[Context | None], bool]:
    """Return whether `read`, given `context`, finds nothing wrong. Success is a model, or a reading with a model."""

    def accepted(context: Context | None) -> bool:
        try:
            found = read(context=context)
        except MetadataValidationError:
            return False
        reading = getattr(found, "reading", found)
        metadata = getattr(reading, "metadata", found)
        return metadata is not None

    return accepted


ENTRY_POINTS: dict[str, Callable[..., object]] = {
    "read_array_metadata_v3": lambda context: zm.read_array_metadata_v3(BAD, context=context),
    "ZarrV3ArrayMetadata": lambda context: zm.ZarrV3ArrayMetadata(BAD, context=context),
    "ZarrV3ArrayMetadata.from_json": lambda context: zm.ZarrV3ArrayMetadata.from_json(
        BAD, context=context
    ),
    "ZarrV3ArrayMetadata.from_key_value": lambda context: zm.ZarrV3ArrayMetadata.from_key_value(
        STORE_A, context=context
    ),
    "ZarrV3ArrayMetadata.create_default": lambda context: zm.ZarrV3ArrayMetadata.create_default(
        shape=(4,), codecs=BAD["codecs"], context=context
    ),
    "ZarrV3ConsolidatedMetadata": lambda context: zm.ZarrV3ConsolidatedMetadata(
        GROUP["consolidated_metadata"], context=context
    ),
    "ZarrV3ConsolidatedMetadata.from_json": lambda context: zm.ZarrV3ConsolidatedMetadata.from_json(
        GROUP["consolidated_metadata"], context=context
    ),
    "read_consolidated_metadata_v3": lambda context: zm.read_consolidated_metadata_v3(
        GROUP["consolidated_metadata"], context=context
    ),
    "read_node_metadata_v3": lambda context: zm.read_node_metadata_v3(BAD, context=context),
    "node_metadata_from_json_v3": lambda context: zm.node_metadata_from_json_v3(
        BAD, context=context
    ),
    "node_metadata_from_key_value_v3": lambda context: zm.node_metadata_from_key_value_v3(
        STORE_A, context=context
    ),
    "read_repaired_node_metadata_v3": lambda context: zm.read_repaired_node_metadata_v3(
        BAD, context=context
    ),
}
"""Each entry point that reads `BAD`, as a document or as an entry of consolidated metadata."""

GROUP_ENTRY_POINTS: dict[str, Callable[..., object]] = {
    "read_group_metadata_v3": lambda context: zm.read_group_metadata_v3(GROUP, context=context),
    "ZarrV3GroupMetadata": lambda context: zm.ZarrV3GroupMetadata(GROUP, context=context),
    "ZarrV3GroupMetadata.from_json": lambda context: zm.ZarrV3GroupMetadata.from_json(
        GROUP, context=context
    ),
    "ZarrV3GroupMetadata.from_key_value": lambda context: zm.ZarrV3GroupMetadata.from_key_value(
        STORE_G, context=context
    ),
    "ZarrV3GroupMetadata.create_default": lambda context: zm.ZarrV3GroupMetadata.create_default(
        consolidated_metadata=GROUP["consolidated_metadata"], context=context
    ),
    "read_node_metadata_v3-group": lambda context: zm.read_node_metadata_v3(GROUP, context=context),
    "node_metadata_from_key_value_v3-group": lambda context: zm.node_metadata_from_key_value_v3(
        STORE_G, context=context
    ),
}
"""Each entry point that reads `GROUP`, which lists `BAD` in its consolidated metadata."""


@pytest.mark.parametrize("read", ENTRY_POINTS.values(), ids=ENTRY_POINTS.keys())
def test_every_entry_point_reads_in_the_scope_it_is_given(read: Callable[..., object]) -> None:
    """Each entry point refuses a gzip level of 12 in the default scope (also selected by `None`), and accepts it in a scope that leaves gzip unclaimed. This shows that each entry point reads in the scope it is given."""
    accepted = _accepts(read)
    assert accepted(EMPTY) is True
    assert accepted(CORE_AND_EXTENSIONS_CONTEXT) is False
    assert accepted(None) is False


@pytest.mark.parametrize("read", GROUP_ENTRY_POINTS.values(), ids=GROUP_ENTRY_POINTS.keys())
def test_a_group_reads_the_same_in_every_scope(read: Callable[..., object]) -> None:
    """A group document has no fields, and a group does not read the documents its consolidated metadata lists. So each group entry point accepts a group that lists `BAD`, in every scope."""
    accepted = _accepts(read)
    assert accepted(EMPTY) is True
    assert accepted(CORE_AND_EXTENSIONS_CONTEXT) is True
    assert accepted(None) is True


def test_the_json_schema_is_written_in_the_scope_it_is_given() -> None:
    """`node_metadata_json_schema_v3` takes `None` for the default scope, and writes a different schema for an empty one."""
    assert node_metadata_json_schema_v3(context=None) == node_metadata_json_schema_v3()
    assert node_metadata_json_schema_v3(context=EMPTY) != node_metadata_json_schema_v3()


@pytest.mark.parametrize(
    "read",
    [*ENTRY_POINTS.values(), *GROUP_ENTRY_POINTS.values()],
    ids=[*ENTRY_POINTS.keys(), *GROUP_ENTRY_POINTS.keys()],
)
def test_error_every_entry_point_refuses_a_scope_of_another_format(
    read: Callable[..., object],
) -> None:
    """A v3 entry point given a v2 scope raises `TypeError`. A scope reads documents of one format, and a v2 scope claims nothing that a v3 document contains."""
    from zarr_metadata.v2.definition import CORE_V2

    with pytest.raises(TypeError, match="format"):
        read(context=CORE_V2)


NOT_JSON = b"{not json"
STORE_READERS: dict[str, Callable[..., object]] = {
    "ZarrV3ArrayMetadata.from_key_value": lambda context: zm.ZarrV3ArrayMetadata.from_key_value(
        {"zarr.json": NOT_JSON}, context=context
    ),
    "ZarrV3GroupMetadata.from_key_value": lambda context: zm.ZarrV3GroupMetadata.from_key_value(
        {"zarr.json": NOT_JSON}, context=context
    ),
    "node_metadata_from_key_value_v3": lambda context: zm.node_metadata_from_key_value_v3(
        {"zarr.json": NOT_JSON}, context=context
    ),
}
"""Each v3 entry point that decodes a store's bytes, given bytes that are not JSON."""


@pytest.mark.parametrize("name", STORE_READERS.keys())
def test_error_a_v3_store_reader_refuses_a_scope_of_another_format_before_decoding(
    name: str,
) -> None:
    """A v3 entry point that reads a store's bytes raises `TypeError` for a v2 scope before it decodes the bytes, so the wrong scope is reported even when the bytes are not JSON. With a v3 scope, the same bytes raise `MetadataValidationError`."""
    from zarr_metadata.v2.definition import CORE_V2

    with pytest.raises(TypeError, match="format"):
        STORE_READERS[name](context=CORE_V2)
    with pytest.raises(MetadataValidationError):
        STORE_READERS[name](context=CORE_AND_EXTENSIONS_CONTEXT)
