"""`unclaimed_fields`: every field of a node that nothing in its scope claims, at any depth, which is what a reader must understand before it opens the node."""

from __future__ import annotations

from typing import Any

import pytest

from zarr_metadata.definition import UnclaimedField
from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3GroupMetadata,
)
from zarr_metadata.v3 import CORE

ARRAY: dict[str, Any] = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
GROUP: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}
V2_ARRAY: dict[str, Any] = dict(ZarrV2ArrayMetadata.create_default(shape=(4,)).to_json())
ZSTD = {"name": "zstd", "configuration": {"level": 1, "checksum": False}}
SHARD = {
    "name": "sharding_indexed",
    "configuration": {
        "chunk_shape": (2,),
        "codecs": ("bytes", ZSTD),
        "index_codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    },
}


@pytest.mark.parametrize(
    ("model", "unclaimed"),
    [
        (ZarrV3ArrayMetadata(ARRAY, context=CORE), set()),
        (ZarrV3ArrayMetadata({**ARRAY, "codecs": ("bytes", ZSTD)}, context=CORE), {("codecs", 1)}),
        (
            ZarrV3ArrayMetadata({**ARRAY, "codecs": (SHARD,)}, context=CORE),
            {("codecs", 0, "configuration", "codecs", 1)},
        ),
        (ZarrV3ArrayMetadata({**ARRAY, "acme": {"x": 1}}, context=CORE), {("acme",)}),
        (ZarrV3ArrayMetadata({**ARRAY, "acme": {"must_understand": False}}, context=CORE), set()),
        (ZarrV3GroupMetadata({**GROUP, "acme": 1}, context=CORE), {("acme",)}),
        (ZarrV2ArrayMetadata({**V2_ARRAY, "compressor": {"id": "acme"}}), {("compressor",)}),
        (ZarrV2GroupMetadata.create_default(), set()),
    ],
    ids=[
        "all-claimed",
        "a-codec",
        "a-codec-inside-a-shard",
        "an-extra-member",
        "a-waived-extra-member",
        "a-group-member",
        "a-v2-codec",
        "a-v2-group",
    ],
)
def test_unclaimed_fields_are_what_a_reader_must_understand(
    model: Any,  # noqa: ANN401 - a model of either format
    unclaimed: set[tuple[str | int, ...]],
) -> None:
    """`unclaimed_fields` maps the location of each field nothing in scope claims to the field, at any depth: a codec inside a shard is located inside the shard's configuration. An extra member that says `"must_understand": false` is left out, as the spec lets a reader ignore it. So a reader can open a node exactly when it is empty."""
    found = model.unclaimed_fields
    assert set(found) == unclaimed
    assert all(isinstance(field, UnclaimedField) for field in found.values())
