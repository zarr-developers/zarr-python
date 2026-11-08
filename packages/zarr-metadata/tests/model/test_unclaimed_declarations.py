"""`unclaimed_declarations`: every declaration of a node that nothing in its scope claims, at any depth, which is what a reader must understand before it opens the node."""

from __future__ import annotations

from typing import Any, NotRequired

import pytest
from typing_extensions import TypedDict

from zarr_metadata import UNSET
from zarr_metadata.definition import UnclaimedDeclaration
from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV2GroupMetadata,
    ZarrV2NodeMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3GroupMetadata,
    ZarrV3NodeMetadata,
    read_node_metadata_v3,
    read_openable_node_metadata_v3,
)
from zarr_metadata.v3 import CORE_CONTEXT
from zarr_metadata.v3.definition import (
    ArrayExtraFieldDefinition,
    GroupExtraFieldDefinition,
    ZarrV3CodecDeclaration,
    ZarrV3Context,
)

ARRAY: dict[str, Any] = dict(ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json())
GROUP: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}
V2_ARRAY: dict[str, Any] = dict(ZarrV2ArrayMetadata.create_default(shape=(4,)).to_json())
ZSTD = {"name": "zstd", "configuration": {"level": 1, "checksum": False}}


class _MirrorConfiguration(TypedDict, closed=True):
    """The value of an `acme.mirror` member: a codec, and the waiver."""

    codec: ZarrV3CodecDeclaration
    must_understand: NotRequired[bool]


MIRRORING = CORE_CONTEXT.union(
    ZarrV3Context(
        array_extra_fields=[
            ArrayExtraFieldDefinition(name="acme.mirror", configuration=_MirrorConfiguration)
        ]
    )
)
"""`CORE_CONTEXT`, with a definition of an `acme.mirror` member that holds a codec."""

GROUP_MIRRORING = CORE_CONTEXT.union(
    ZarrV3Context(
        group_extra_fields=[
            GroupExtraFieldDefinition(name="acme.mirror", configuration=_MirrorConfiguration)
        ]
    )
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
def test_unclaimed_declarations_are_what_a_reader_must_understand(
    model: Any,  # noqa: ANN401 - a model of either format
    unclaimed: set[tuple[str | int, ...]],
) -> None:
    """`unclaimed_declarations` maps the location of each declaration nothing in scope claims to the declaration, at any depth: a codec inside a shard is located inside the shard's configuration. An extra field that says `"must_understand": false` is left out with every declaration inside it, claimed or not, as the spec lets a reader ignore it. So a reader can open a node exactly when it is empty."""
    found = model.unclaimed_declarations
    assert set(found) == unclaimed
    assert all(isinstance(declaration, UnclaimedDeclaration) for declaration in found.values())


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
    """A scope that claims more of a document reads a model that refines the model of a scope that claims less, and its `unclaimed_declarations` lie at or inside those of the model it refines, for an array and for a group. Here `CORE_CONTEXT` leaves an unwaived `acme.mirror` member unclaimed as a whole; the scope that claims the member leaves at most the codec inside it, which lies inside the member, and a waived member is left out by both."""
    claimed_less = kind({**document, "acme.mirror": member}, context=CORE_CONTEXT)
    claimed_more = kind({**document, "acme.mirror": member}, context=context)
    assert claimed_more.refines(claimed_less)
    assert set(claimed_less.unclaimed_declarations) == wider
    assert set(claimed_more.unclaimed_declarations) == narrower
    assert all(
        any(loc[: len(outer)] == outer for outer in claimed_less.unclaimed_declarations)
        for loc in claimed_more.unclaimed_declarations
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


@pytest.mark.parametrize(
    ("document", "context", "unclaimed", "problems"),
    [
        (ARRAY, CORE_CONTEXT, set(), set()),
        ({**ARRAY, "codecs": ("bytes", ZSTD)}, CORE_CONTEXT, {("codecs", 1)}, set()),
        (
            {**ARRAY, "codecs": (SHARD,)},
            CORE_CONTEXT,
            {("codecs", 0, "configuration", "codecs", 1)},
            set(),
        ),
        ({**ARRAY, "acme": 1}, CORE_CONTEXT, {("acme",)}, set()),
        ({**ARRAY, "acme": {"must_understand": False}}, CORE_CONTEXT, set(), set()),
        ({**GROUP, "acme": {"name": "acme"}}, CORE_CONTEXT, {("acme",)}, set()),
        ({**GROUP, "acme": {"must_understand": False}}, CORE_CONTEXT, set(), set()),
        (
            {**ARRAY, "acme.mirror": {"codec": {"name": "acme.zip"}}},
            MIRRORING,
            {("acme.mirror", "codec")},
            set(),
        ),
        (
            {**ARRAY, "acme.mirror": {"codec": {"name": "acme.zip"}, "must_understand": False}},
            MIRRORING,
            set(),
            set(),
        ),
        (
            {**ARRAY, "fill_value": "x", "codecs": ("bytes", ZSTD)},
            CORE_CONTEXT,
            {("codecs", 1)},
            {("fill_value",)},
        ),
        ({**GROUP, "node_type": "acme"}, CORE_CONTEXT, set(), {("node_type",)}),
        ("not a node", CORE_CONTEXT, set(), {()}),
    ],
    ids=[
        "all-claimed",
        "a-codec",
        "a-codec-inside-a-shard",
        "an-extra-member",
        "a-waived-extra-member",
        "a-group-member",
        "a-waived-group-member",
        "a-codec-inside-a-claimed-member",
        "a-codec-inside-a-claimed-member-that-waives",
        "with-other-problems",
        "an-unknown-node-type",
        "not-an-object",
    ],
)
def test_read_openable_node_metadata_v3_reports_what_a_reader_must_understand(
    document: object,
    context: Any,  # noqa: ANN401 - a v3 scope
    unclaimed: set[tuple[str | int, ...]],
    problems: set[tuple[str | int, ...]],
) -> None:
    """`read_openable_node_metadata_v3` keeps every problem `read_node_metadata_v3` finds and adds one of kind `unclaimed_declaration` at the location of each declaration `unclaimed_declarations` lists, with what the document writes there as its input, whether or not the document has other problems. It gives a model exactly when there are no problems, and the model is the one `read_node_metadata_v3` gives."""
    plain = read_node_metadata_v3(document, context=context)
    reading = read_openable_node_metadata_v3(document, context=context)
    assert reading.problems[: len(plain.problems)] == plain.problems
    assert {problem.loc for problem in plain.problems} == problems
    added = reading.problems[len(plain.problems) :]
    assert {problem.loc for problem in added} == unclaimed
    assert all(problem.kind == "unclaimed_declaration" for problem in added)
    assert all(problem.input is not UNSET for problem in added)
    if plain.metadata is not None:
        assert set(plain.metadata.unclaimed_declarations) == unclaimed
    if len(reading.problems) == 0:
        assert reading.metadata == plain.metadata
        assert reading.metadata is not None
    else:
        assert reading.metadata is None
