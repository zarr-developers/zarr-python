"""Repairing v3 metadata a known writer bug made invalid, before a strict read."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from zarr_metadata.model import (
    MetadataValidationError,
    ZarrV2ConsolidatedMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3GroupMetadata,
    read_node_metadata_v3,
    read_repaired_consolidated_metadata_v2,
    read_repaired_node_metadata_v3,
    repair_consolidated_metadata_v2,
    repair_node_metadata_v3,
)

ARRAY: dict[str, Any] = dict(ZarrV3ArrayMetadata.create_default(shape=(0, 3)).to_json())
GROUP: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}


def _grid(*chunk_shape: int) -> dict[str, Any]:
    return {"name": "regular", "configuration": {"chunk_shape": chunk_shape}}


def _consolidated(**metadata: object) -> dict[str, Any]:
    envelope = {"kind": "inline", "must_understand": False, "metadata": metadata}
    return {**GROUP, "consolidated_metadata": envelope}


ZERO_CHUNK = {**ARRAY, "chunk_grid": _grid(0, 3)}
NULL_CONSOLIDATED = {**GROUP, "consolidated_metadata": None}


@pytest.mark.parametrize(
    ("value", "repaired", "repairs"),
    [
        (
            ZERO_CHUNK,
            {**ARRAY, "chunk_grid": _grid(1, 3)},
            [(("chunk_grid", "configuration", "chunk_shape", 0), "zero_chunk_length")],
        ),
        (
            {**ARRAY, "shape": (0, 0), "chunk_grid": _grid(0, 0)},
            {**ARRAY, "shape": (0, 0), "chunk_grid": _grid(1, 1)},
            [
                (("chunk_grid", "configuration", "chunk_shape", 0), "zero_chunk_length"),
                (("chunk_grid", "configuration", "chunk_shape", 1), "zero_chunk_length"),
            ],
        ),
        (NULL_CONSOLIDATED, GROUP, [(("consolidated_metadata",), "null_consolidated_metadata")]),
        (
            _consolidated(a=ZERO_CHUNK, b=NULL_CONSOLIDATED),
            _consolidated(a={**ARRAY, "chunk_grid": _grid(1, 3)}, b=GROUP),
            [
                (
                    ("consolidated_metadata", "metadata", "a", "chunk_grid", "configuration")
                    + ("chunk_shape", 0),
                    "zero_chunk_length",
                ),
                (
                    ("consolidated_metadata", "metadata", "b", "consolidated_metadata"),
                    "null_consolidated_metadata",
                ),
            ],
        ),
        # No writer's bug: left for the strict read to judge.
        (ARRAY, ARRAY, []),
        ({**ARRAY, "chunk_grid": _grid(3, 0)}, {**ARRAY, "chunk_grid": _grid(3, 0)}, []),
        ({**ARRAY, "chunk_grid": _grid(0)}, {**ARRAY, "chunk_grid": _grid(0)}, []),
        (
            {key: item for key, item in ZERO_CHUNK.items() if key != "data_type"},
            {key: item for key, item in ARRAY.items() if key != "data_type"}
            | {"chunk_grid": _grid(1, 3)},
            [(("chunk_grid", "configuration", "chunk_shape", 0), "zero_chunk_length")],
        ),
        ({**GROUP, "consolidated_metadata": 0}, {**GROUP, "consolidated_metadata": 0}, []),
        ([ZERO_CHUNK], [ZERO_CHUNK], []),
    ],
    ids=[
        "zero-chunk",
        "zero-chunks",
        "null-consolidated",
        "inside-consolidated",
        "valid",
        "zero-chunk-on-a-full-dimension",
        "zero-chunk-of-another-rank",
        "zero-chunk-without-a-data-type",
        "consolidated-of-another-type",
        "not-a-document",
    ],
)
def test_repair_undoes_each_known_writer_bug(
    value: object, repaired: object, repairs: list[tuple[tuple[str | int, ...], str]]
) -> None:
    """Each known writer bug in a document, its consolidated documents' too, is undone and said where; anything else is left as it is, for the strict read to report, and the document handed in is not changed."""
    before = copy.deepcopy(value)
    document, made = repair_node_metadata_v3(value)
    assert document == repaired
    assert [(repair.loc, repair.kind) for repair in made] == repairs
    assert value == before
    if len(repairs) == 0:
        assert document is value


@pytest.mark.parametrize(
    ("value", "model"),
    [
        (ZERO_CHUNK, ZarrV3ArrayMetadata.from_json({**ARRAY, "chunk_grid": _grid(1, 3)})),
        (NULL_CONSOLIDATED, ZarrV3GroupMetadata.from_json(GROUP)),
        (ARRAY, ZarrV3ArrayMetadata.from_json(ARRAY)),
    ],
    ids=["zero-chunk", "null-consolidated", "valid"],
)
def test_read_repaired_reads_what_the_strict_read_refuses(value: object, model: object) -> None:
    """A document a writer bug made invalid is refused by `read_node_metadata_v3` and read by `read_repaired_node_metadata_v3`, as the strict read reads the repaired document; a valid one reads the same either way."""
    repaired = read_repaired_node_metadata_v3(value)
    assert repaired.reading.problems == ()
    assert repaired.reading.metadata == model
    assert (len(read_node_metadata_v3(value).problems) == 0) is (len(repaired.repairs) == 0)


def test_read_repaired_reports_what_no_repair_applies_to() -> None:
    """A problem no repair undoes is reported as the strict read reports it, beside the repairs that were made."""
    value = {key: item for key, item in ZERO_CHUNK.items() if key != "data_type"}
    repaired = read_repaired_node_metadata_v3(value)
    assert [repair.kind for repair in repaired.repairs] == ["zero_chunk_length"]
    assert [(problem.loc, problem.kind) for problem in repaired.reading.problems] == [
        (("data_type",), "missing_key")
    ]
    assert repaired.reading.metadata is None


# --- v2 ---------------------------------------------------------------------

ZGROUP_FROM_ZARR3: dict[str, Any] = {
    "zarr_format": 2,
    "consolidated_metadata": {"metadata": {}, "must_understand": False, "kind": "inline"},
}
ZMETADATA_FROM_ZARR3: dict[str, Any] = {
    "zarr_consolidated_format": 1,
    "metadata": {
        ".zgroup": {"zarr_format": 2},
        "b/.zgroup": ZGROUP_FROM_ZARR3,
        "b/.zattrs": {"x": 1},
    },
}
ZMETADATA_CLEAN: dict[str, Any] = {
    "zarr_consolidated_format": 1,
    "metadata": {
        ".zgroup": {"zarr_format": 2},
        "b/.zgroup": {"zarr_format": 2},
        "b/.zattrs": {"x": 1},
    },
}


@pytest.mark.parametrize(
    ("value", "repaired", "repairs"),
    [
        (
            ZMETADATA_FROM_ZARR3,
            ZMETADATA_CLEAN,
            [
                (
                    ("metadata", "b/.zgroup", "consolidated_metadata"),
                    "consolidated_metadata_in_zgroup_entry",
                )
            ],
        ),
        (ZMETADATA_CLEAN, ZMETADATA_CLEAN, []),
        (
            {
                "zarr_consolidated_format": 1,
                "metadata": {"b/.zgroup": {"zarr_format": 2, "consolidated_metadata": 3}},
            },
            {
                "zarr_consolidated_format": 1,
                "metadata": {"b/.zgroup": {"zarr_format": 2, "consolidated_metadata": 3}},
            },
            [],
        ),
        (3, 3, []),
    ],
    ids=["zarr-3-zgroup-entry", "clean", "not-the-bug", "not-a-document"],
)
def test_repair_v2_undoes_the_consolidated_metadata_zarr_3_writes_into_a_zgroup_entry(
    value: object, repaired: object, repairs: list[tuple[tuple[str | int, ...], str]]
) -> None:
    """zarr-python 3.x writes a `consolidated_metadata` member into each non-root `.zgroup` entry of a `.zmetadata`, which the v2 group document does not take; `repair_consolidated_metadata_v2` removes it and says where; anything else is left as it is, and the document handed in is not changed."""
    before = copy.deepcopy(value)
    document, made = repair_consolidated_metadata_v2(value)
    assert document == repaired
    assert [(repair.loc, repair.kind) for repair in made] == repairs
    assert value == before
    if len(repairs) == 0:
        assert document is value


def test_read_repaired_consolidated_v2_reads_what_the_strict_read_refuses() -> None:
    """The strict `ZarrV2ConsolidatedMetadata` refuses the zarr-python 3.x `.zmetadata` at the entry's member; `read_repaired_consolidated_metadata_v2` reads the repaired document, holds its model, and lists the repairs."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ConsolidatedMetadata(ZMETADATA_FROM_ZARR3)
    assert [p.loc for p in raised.value.problems] == [
        ("metadata", "b/.zgroup", "consolidated_metadata")
    ]
    repaired = read_repaired_consolidated_metadata_v2(ZMETADATA_FROM_ZARR3)
    assert repaired.problems == ()
    assert repaired.metadata == ZarrV2ConsolidatedMetadata(ZMETADATA_CLEAN)
    assert [r.kind for r in repaired.repairs] == ["consolidated_metadata_in_zgroup_entry"]
    broken = read_repaired_consolidated_metadata_v2(
        {
            **ZMETADATA_FROM_ZARR3,
            "metadata": {**ZMETADATA_FROM_ZARR3["metadata"], "b/.zattrs": {1: "x"}},
        }
    )
    assert broken.metadata is None
    assert [p.loc for p in broken.problems] == [("metadata", "b/.zattrs")]
    assert [r.kind for r in broken.repairs] == ["consolidated_metadata_in_zgroup_entry"]
