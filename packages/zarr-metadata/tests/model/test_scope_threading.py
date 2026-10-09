"""Every entry point reads in the scope it is given, and in `CORE_AND_EXTENSIONS` when given none."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

import zarr_metadata.model as zm
from zarr_metadata.model import (
    ZarrV3ArrayMetadata,
)
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS,
    Context,
)

if TYPE_CHECKING:
    from collections.abc import Callable

EMPTY = Context.of()
BASE: dict[str, Any] = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
BAD: dict[str, Any] = {
    **BASE,
    "codecs": (*BASE["codecs"], {"name": "gzip", "configuration": {"level": 12}}),
}
"""An array the default scope refuses -- a gzip level of 12 -- and an empty scope leaves unjudged."""
GROUP: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "group",
    "consolidated_metadata": {"kind": "inline", "must_understand": False, "metadata": {"a": BAD}},
}
STORE_A = {"zarr.json": json.dumps(BAD).encode()}
STORE_G = {"zarr.json": json.dumps(GROUP).encode()}


def _accepts(read: Callable[..., object]) -> Callable[[Context | None], bool]:
    """Whether `read`, given `context`, finds nothing wrong: True for a model, a reading with a model, an empty problem tuple, or a guard saying yes."""

    def accepted(context: Context | None) -> bool:
        try:
            found = read(context=context)
        except zm.MetadataValidationError:
            return False
        if isinstance(found, tuple):
            return len(found) == 0
        if isinstance(found, bool):
            return found
        reading = getattr(found, "reading", found)
        metadata = getattr(reading, "metadata", found)
        return metadata is not None

    return accepted


ENTRY_POINTS: dict[str, Callable[..., object]] = {
    "validate_array_metadata_v3": lambda context: zm.validate_array_metadata_v3(
        BAD, context=context
    ),
    "is_array_metadata_v3": lambda context: zm.is_array_metadata_v3(
        zm.parse_array_metadata_v3(BAD, context=EMPTY), context=context
    ),
    "parse_array_metadata_v3": lambda context: zm.parse_array_metadata_v3(BAD, context=context),
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
    "validate_group_metadata_v3": lambda context: zm.validate_group_metadata_v3(
        GROUP, context=context
    ),
    "is_group_metadata_v3": lambda context: zm.is_group_metadata_v3(
        zm.parse_group_metadata_v3(GROUP, context=EMPTY), context=context
    ),
    "parse_group_metadata_v3": lambda context: zm.parse_group_metadata_v3(GROUP, context=context),
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
    "ZarrV3ConsolidatedMetadata": lambda context: zm.ZarrV3ConsolidatedMetadata(
        GROUP["consolidated_metadata"], context=context
    ),
    "ZarrV3ConsolidatedMetadata.from_json": lambda context: zm.ZarrV3ConsolidatedMetadata.from_json(
        GROUP["consolidated_metadata"], context=context
    ),
    "read_node_metadata_v3": lambda context: zm.read_node_metadata_v3(GROUP, context=context),
    "validate_node_metadata_v3": lambda context: zm.validate_node_metadata_v3(
        GROUP, context=context
    ),
    "node_metadata_from_json_v3": lambda context: zm.node_metadata_from_json_v3(
        GROUP, context=context
    ),
    "node_metadata_from_key_value_v3": lambda context: zm.node_metadata_from_key_value_v3(
        STORE_G, context=context
    ),
    "read_repaired_node_metadata_v3": lambda context: zm.read_repaired_node_metadata_v3(
        GROUP, context=context
    ),
}


@pytest.mark.parametrize("read", ENTRY_POINTS.values(), ids=ENTRY_POINTS.keys())
def test_every_entry_point_reads_in_the_scope_it_is_given(read: Callable[..., object]) -> None:
    """Each entry point refuses a gzip level of 12 in the default scope, which `None` names too, and accepts it in a scope that leaves gzip unclaimed: the scope it is given is the scope it reads in."""
    accepted = _accepts(read)
    assert accepted(EMPTY) is True
    assert accepted(CORE_AND_EXTENSIONS) is False
    assert accepted(None) is False


def test_the_json_schema_is_written_in_the_scope_it_is_given() -> None:
    """`node_metadata_json_schema_v3` takes `None` for the default scope, and writes a different schema for an empty one."""
    assert zm.node_metadata_json_schema_v3(context=None) == zm.node_metadata_json_schema_v3()
    assert zm.node_metadata_json_schema_v3(context=EMPTY) != zm.node_metadata_json_schema_v3()


@pytest.mark.parametrize("read", ENTRY_POINTS.values(), ids=ENTRY_POINTS.keys())
def test_error_every_entry_point_refuses_a_scope_of_another_format(
    read: Callable[..., object],
) -> None:
    """A v3 entry point given a v2 scope raises `TypeError`: a scope reads documents of one format, and a v2 scope claims nothing a v3 document writes."""
    from zarr_metadata.v2.definition import CORE_V2

    with pytest.raises(TypeError, match="format"):
        read(context=CORE_V2)
