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
from zarr_metadata.v3 import CORE_CONTEXT
from zarr_metadata.v3.definition import (
    ArrayExtraFieldDefinition,
    GroupExtraFieldDefinition,
    ZarrV3CodecField,
)

ARRAY: dict[str, Any] = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
GROUP: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}
V2_ARRAY: dict[str, Any] = dict(ZarrV2ArrayMetadata.create_default(shape=(4,)).to_json())
ZSTD = {"name": "zstd", "configuration": {"level": 1, "checksum": False}}


class _MirrorConfiguration(TypedDict, closed=True):
    """The value of an `acme.mirror` member: a codec, and the waiver."""

    codec: ZarrV3CodecField
    must_understand: NotRequired[bool]


MIRRORING = CORE_CONTEXT.update(
    array_extra_fields=[
        *CORE_CONTEXT.array_extra_fields.values(),
        ArrayExtraFieldDefinition(name="acme.mirror", configuration=_MirrorConfiguration),
    ]
)
"""`CORE_CONTEXT`, with a definition of an `acme.mirror` member that holds a codec."""

GROUP_MIRRORING = CORE_CONTEXT.update(
    group_extra_fields=[
        *CORE_CONTEXT.group_extra_fields.values(),
        GroupExtraFieldDefinition(name="acme.mirror", configuration=_MirrorConfiguration),
    ]
)
"""`CORE_CONTEXT`, with a definition of an `acme.mirror` member of a group that holds a codec."""

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
        (ZarrV3ArrayMetadata(ARRAY, context=CORE_CONTEXT), set()),
        (
            ZarrV3ArrayMetadata({**ARRAY, "codecs": ("bytes", ZSTD)}, context=CORE_CONTEXT),
            {("codecs", 1)},
        ),
        (
            ZarrV3ArrayMetadata({**ARRAY, "codecs": (SHARD,)}, context=CORE_CONTEXT),
            {("codecs", 0, "configuration", "codecs", 1)},
        ),
        (ZarrV3ArrayMetadata({**ARRAY, "acme": {"x": 1}}, context=CORE_CONTEXT), {("acme",)}),
        (
            ZarrV3ArrayMetadata(
                {**ARRAY, "acme": {"must_understand": False}}, context=CORE_CONTEXT
            ),
            set(),
        ),
        (ZarrV3GroupMetadata({**GROUP, "acme": 1}, context=CORE_CONTEXT), {("acme",)}),
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
        (
            ZarrV3GroupMetadata(
                {**GROUP, "acme.mirror": {"codec": {"name": "acme.zip"}}}, context=GROUP_MIRRORING
            ),
            {("acme.mirror", "codec")},
        ),
        (
            ZarrV3GroupMetadata(
                {**GROUP, "acme.mirror": {"codec": {"name": "acme.zip"}, "must_understand": False}},
                context=GROUP_MIRRORING,
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
        "a-codec-inside-a-claimed-group-member",
        "a-codec-inside-a-claimed-group-member-that-waives",
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


@pytest.mark.parametrize(
    ("kind", "document", "context"),
    [(ZarrV3ArrayMetadata, ARRAY, MIRRORING), (ZarrV3GroupMetadata, GROUP, GROUP_MIRRORING)],
    ids=["array", "group"],
)
@pytest.mark.parametrize(
    ("member", "wider", "narrower"),
    [
        (
            {"codec": {"name": "gzip", "configuration": {"level": 1}}},
            {("acme.mirror",)},
            set(),
        ),
        ({"codec": {"name": "acme.zip"}}, {("acme.mirror",)}, {("acme.mirror", "codec")}),
        ({"codec": {"name": "acme.zip"}, "must_understand": False}, set(), set()),
    ],
    ids=["a-claimed-codec", "an-unclaimed-codec", "waived"],
)
def test_a_model_that_refines_another_has_no_more_to_understand(
    kind: Any,  # noqa: ANN401 - a v3 node model class
    document: dict[str, Any],
    context: Any,  # noqa: ANN401 - a v3 scope
    member: dict[str, Any],
    wider: set[tuple[str | int, ...]],
    narrower: set[tuple[str | int, ...]],
) -> None:
    """A scope that claims more of a document reads a model that refines the model of a scope that claims less, and its `unclaimed_fields` lie at or inside those of the model it refines, for an array and for a group. Here `CORE_CONTEXT` leaves an unwaived `acme.mirror` member unclaimed as a whole; the scope that claims the member leaves at most the codec inside it, which lies inside the member, and a waived member is left out by both."""
    claimed_less = kind({**document, "acme.mirror": member}, context=CORE_CONTEXT)
    claimed_more = kind({**document, "acme.mirror": member}, context=context)
    assert claimed_more.refines(claimed_less)
    assert set(claimed_less.unclaimed_fields) == wider
    assert set(claimed_more.unclaimed_fields) == narrower
    assert all(
        any(loc[: len(outer)] == outer for outer in claimed_less.unclaimed_fields)
        for loc in claimed_more.unclaimed_fields
    )


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
