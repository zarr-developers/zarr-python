"""`unclaimed_fields`: every field of a node that nothing in its scope claims, at any depth, which is what a reader must understand before it opens the node."""

from __future__ import annotations

from typing import Any, NotRequired

import pytest
from typing_extensions import TypedDict

from zarr_metadata.definition import UnclaimedField
from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV2GroupMetadata,
    ZarrV2NodeMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3GroupMetadata,
    ZarrV3NodeMetadata,
)
from zarr_metadata.v3 import CORE
from zarr_metadata.v3.definition import ArrayExtraFieldDefinition, ZarrV3CodecField

ARRAY: dict[str, Any] = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
GROUP: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}
V2_ARRAY: dict[str, Any] = dict(ZarrV2ArrayMetadata.create_default(shape=(4,)).to_json())
ZSTD = {"name": "zstd", "configuration": {"level": 1, "checksum": False}}


class _MirrorConfiguration(TypedDict, closed=True):
    """The value of an `acme.mirror` member: a codec, and the waiver."""

    codec: ZarrV3CodecField
    must_understand: NotRequired[bool]


MIRRORING = CORE.update(
    array_extra_fields=[
        *CORE.array_extra_fields.values(),
        ArrayExtraFieldDefinition(name="acme.mirror", configuration=_MirrorConfiguration),
    ]
)
"""`CORE`, with a definition of an `acme.mirror` member that holds a codec."""

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
        (
            ZarrV3ArrayMetadata(
                {**ARRAY, "acme.mirror": {"codec": {"name": "acme.zip"}}}, context=MIRRORING
            ),
            {("acme.mirror", "codec")},
        ),
        (
            ZarrV3ArrayMetadata(
                {**ARRAY, "acme.mirror": {"codec": {"name": "acme.zip"}, "must_understand": False}},
                context=MIRRORING,
            ),
            set(),
        ),
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
        "a-codec-inside-a-claimed-member",
        "a-codec-inside-a-claimed-member-that-waives",
    ],
)
def test_unclaimed_fields_are_what_a_reader_must_understand(
    model: Any,  # noqa: ANN401 - a model of either format
    unclaimed: set[tuple[str | int, ...]],
) -> None:
    """`unclaimed_fields` maps the location of each field nothing in scope claims to the field, at any depth: a codec inside a shard is located inside the shard's configuration. An extra member that says `"must_understand": false` is left out with every field inside it, claimed or not, as the spec lets a reader ignore it. So a reader can open a node exactly when it is empty."""
    found = model.unclaimed_fields
    assert set(found) == unclaimed
    assert all(isinstance(field, UnclaimedField) for field in found.values())


def test_node_type_tells_a_type_checker_which_model_a_node_is() -> None:
    """`node_type` and `zarr_format` are literal types, so branching on them narrows a union of node models for a type checker as well as at run time. pyright checks this test: it reads `shape` only on arrays, and `consolidated()` only on a v3 group."""
    nodes: list[ZarrV2NodeMetadata | ZarrV3NodeMetadata] = [
        ZarrV3ArrayMetadata.create_default(shape=(4,)),
        ZarrV3GroupMetadata.create_default(),
        ZarrV2ArrayMetadata.create_default(shape=(4,)),
        ZarrV2GroupMetadata.create_default(),
    ]
    shapes = [node.shape for node in nodes if node.node_type == "array"]
    assert shapes == [(4,), (4,)]
    for node in nodes:
        if node.zarr_format == 3 and node.node_type == "group":
            assert node.consolidated() is None
