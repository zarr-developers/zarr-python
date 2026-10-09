"""Every v2 entry point reads in the scope it is given, `CORE_V2` when given none, and refuses a scope of another format."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

import zarr_metadata.model as zm
from zarr_metadata.model import ZarrV2ArrayMetadata
from zarr_metadata.v2.definition import CORE_V2, Context, resolve_codec_v2, resolve_dtype_v2
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS

if TYPE_CHECKING:
    from collections.abc import Callable

EMPTY = Context.of()
BASE: dict[str, Any] = dict(ZarrV2ArrayMetadata.create_default(shape=(4,), chunks=(2,)).to_json())
BAD: dict[str, Any] = {**BASE, "compressor": {"id": "gzip", "level": "x"}}
"""An array the v2 scope refuses -- a gzip level that is no integer -- and an empty scope leaves unjudged."""
GROUP: dict[str, Any] = {"zarr_format": 2}
CONSOLIDATED: dict[str, Any] = {
    "zarr_consolidated_format": 1,
    "metadata": {".zgroup": GROUP, "a/.zarray": BAD},
}
STORE_A = {".zarray": json.dumps(BAD).encode()}
STORE_G = {".zgroup": json.dumps(GROUP).encode()}
STORE_C = {".zmetadata": json.dumps(CONSOLIDATED).encode()}


def _accepts(read: Callable[..., object]) -> Callable[[Context | None], bool]:
    """Whether `read`, given `context`, finds nothing wrong: True for a model, a reading with a model, an empty problem tuple, a guard saying yes, or a field read or left unclaimed."""

    def accepted(context: Context | None) -> bool:
        try:
            found = read(context=context)
        except zm.MetadataValidationError:
            return False
        if isinstance(found, tuple) and len(found) == 2 and isinstance(found[1], tuple):
            return len(found[1]) == 0  # a resolved field and its problems
        if isinstance(found, tuple):
            return len(found) == 0
        if isinstance(found, bool):
            return found
        reading = getattr(found, "reading", found)
        metadata = getattr(reading, "metadata", found)
        return metadata is not None

    return accepted


ENTRY_POINTS: dict[str, Callable[..., object]] = {
    "validate_array_metadata_v2": lambda context: zm.validate_array_metadata_v2(
        BAD, context=context
    ),
    "is_array_metadata_v2": lambda context: zm.is_array_metadata_v2(
        zm.parse_array_metadata_v2(BAD, context=EMPTY), context=context
    ),
    "parse_array_metadata_v2": lambda context: zm.parse_array_metadata_v2(BAD, context=context),
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
    "validate_group_metadata_v2": lambda context: zm.validate_group_metadata_v2(
        GROUP, context=context
    ),
    "is_group_metadata_v2": lambda context: zm.is_group_metadata_v2(GROUP, context=context),
    "parse_group_metadata_v2": lambda context: zm.parse_group_metadata_v2(GROUP, context=context),
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
    "resolve_dtype_v2": lambda context: resolve_dtype_v2("<u1", context=context),
    "resolve_codec_v2": lambda context: resolve_codec_v2(BAD["compressor"], context=context),
}
"""Each v2 entry point, reading a document that holds the refused compressor where it can hold one."""

REFUSES_BAD = frozenset(
    name
    for name in ENTRY_POINTS
    if "Group" not in name and name != "resolve_dtype_v2" and "group" not in name
)
"""The entry points whose document holds the refused compressor: the rest accept their document in every scope."""


@pytest.mark.parametrize("name", ENTRY_POINTS.keys())
def test_every_v2_entry_point_reads_in_the_scope_it_is_given(name: str) -> None:
    """Each v2 entry point accepts its document in an empty scope, and refuses the gzip level that is no integer in `CORE_V2`, which `None` names too, when the document holds it: the scope it is given is the scope it reads in."""
    accepted = _accepts(ENTRY_POINTS[name])
    assert accepted(EMPTY) is True
    assert accepted(CORE_V2) is (name not in REFUSES_BAD)
    assert accepted(None) is (name not in REFUSES_BAD)


@pytest.mark.parametrize("name", ENTRY_POINTS.keys())
def test_error_every_v2_entry_point_refuses_a_scope_of_another_format(name: str) -> None:
    """A v2 entry point given a v3 scope raises `TypeError`: a scope reads documents of one format, and a v3 scope claims nothing a v2 document writes."""
    with pytest.raises(TypeError, match="format"):
        ENTRY_POINTS[name](context=CORE_AND_EXTENSIONS)
