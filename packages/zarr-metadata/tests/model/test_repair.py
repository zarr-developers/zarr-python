"""Tests for repairing v2 and v3 metadata that a known writer bug made invalid, before a strict read."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3GroupMetadata,
    read_node_metadata_v2,
    read_node_metadata_v3,
    read_repaired_consolidated_metadata_v2,
    read_repaired_node_metadata_v2,
    read_repaired_node_metadata_v3,
    repair_consolidated_metadata_v2,
    repair_node_metadata_v2,
    repair_node_metadata_v3,
)
from zarr_metadata.problems import MetadataValidationError

ARRAY: dict[str, Any] = dict(ZarrV3ArrayMetadata.create_default(shape=(0, 3)).to_json())
GROUP: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}


def _grid(*chunk_shape: int) -> dict[str, Any]:
    return {"name": "regular", "configuration": {"chunk_shape": chunk_shape}}


def _consolidated(**metadata: object) -> dict[str, Any]:
    envelope = {"kind": "inline", "must_understand": False, "metadata": metadata}
    return {**GROUP, "consolidated_metadata": envelope}


ZERO_CHUNK = {**ARRAY, "chunk_grid": _grid(0, 3)}
NULL_CONSOLIDATED = {**GROUP, "consolidated_metadata": None}
FLOAT64: dict[str, Any] = {**ARRAY, "data_type": "float64", "fill_value": 0.0}
COMPLEX128: dict[str, Any] = {**ARRAY, "data_type": "complex128", "fill_value": (0.0, 0.0)}
NAN_INT: dict[str, Any] = {**ARRAY, "fill_value": math.nan}
NAN_FLOAT64: dict[str, Any] = {**FLOAT64, "fill_value": math.nan}


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
        (
            _consolidated(a={**FLOAT64, "fill_value": math.nan}),
            _consolidated(a={**FLOAT64, "fill_value": "NaN"}),
            [(("consolidated_metadata", "metadata", "a", "fill_value"), "non_finite_fill_value")],
        ),
        (
            _consolidated(a={**COMPLEX128, "fill_value": (math.inf, 0.0)}),
            _consolidated(a={**COMPLEX128, "fill_value": ("Infinity", 0.0)}),
            [
                (
                    ("consolidated_metadata", "metadata", "a", "fill_value", 0),
                    "non_finite_fill_value",
                )
            ],
        ),
        (
            _consolidated(b={**FLOAT64, "fill_value": -math.inf}),
            _consolidated(b={**FLOAT64, "fill_value": "-Infinity"}),
            [
                (
                    ("consolidated_metadata", "metadata", "b", "fill_value"),
                    "non_finite_fill_value",
                )
            ],
        ),
        # Not a known writer bug: left for the strict read to check.
        (ARRAY, ARRAY, []),
        (NAN_INT, NAN_INT, []),
        (NAN_FLOAT64, NAN_FLOAT64, []),
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
        "non-finite-float-fill",
        "non-finite-complex-fill",
        "non-finite-fill-inside-consolidated",
        "valid",
        "non-finite-fill-of-an-integer-type",
        "non-finite-fill-outside-a-listing",
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
    """Each known writer bug in a document, including in its consolidated documents, is undone, and each repair reports its location: a zero chunk length along an empty dimension, a null `consolidated_metadata`, and a float or complex fill value that zarr-python 3.0.0 to 3.0.6 wrote as a bare `NaN` or infinity in a listing. Those versions wrote a node's own `zarr.json` correctly, so such a fill value there is no known writer bug. Anything else is left unchanged for the strict read to report. The input document is not modified."""
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
    """A document made invalid by a writer bug is refused by `read_node_metadata_v3`. `read_repaired_node_metadata_v3` reads it, with the same result as a strict read of the repaired document. A valid document reads the same either way."""
    repaired = read_repaired_node_metadata_v3(value)
    assert repaired.reading.problems == ()
    assert repaired.reading.metadata == model
    assert (len(read_node_metadata_v3(value).problems) == 0) is (len(repaired.repairs) == 0)


def test_read_repaired_reports_what_no_repair_applies_to() -> None:
    """A problem that no repair fixes is reported the same way the strict read reports it, alongside the repairs that were made."""
    value = {key: item for key, item in ZERO_CHUNK.items() if key != "data_type"}
    repaired = read_repaired_node_metadata_v3(value)
    assert [repair.kind for repair in repaired.repairs] == ["zero_chunk_length"]
    assert [(problem.loc, problem.kind) for problem in repaired.reading.problems] == [
        (("data_type",), "missing_key")
    ]
    assert repaired.reading.metadata is None


# --- v2 ---------------------------------------------------------------------

# The `.zarray` that zarr-python 3.0.0 to 3.0.6 wrote for a `<f4` array whose fill value is NaN.
NAN_ZARRAY_BYTES = (
    b'{"shape": [2], "chunks": [2], "fill_value": NaN, "order": "C", "filters": null, '
    b'"dimension_separator": ".", "compressor": {"id": "zstd", "level": 0}, '
    b'"zarr_format": 2, "dtype": "<f4"}'
)
NAN_ZARRAY: dict[str, Any] = json.loads(NAN_ZARRAY_BYTES)
NAMED_ZARRAY: dict[str, Any] = {**NAN_ZARRAY, "fill_value": "NaN"}
COMPLEX_ZARRAY: dict[str, Any] = {**NAN_ZARRAY, "dtype": "<c8"}


def _with_fill(zarray: dict[str, Any], fill_value: object) -> dict[str, object]:
    return {".zarray": {**zarray, "fill_value": fill_value}, ".zattrs": {}}


@pytest.mark.parametrize(
    ("files", "repaired", "repairs"),
    [
        (
            _with_fill(NAN_ZARRAY, math.nan),
            _with_fill(NAN_ZARRAY, "NaN"),
            [((".zarray", "fill_value"), "non_finite_fill_value")],
        ),
        (
            _with_fill({**NAN_ZARRAY, "dtype": ">f8"}, -math.inf),
            _with_fill({**NAN_ZARRAY, "dtype": ">f8"}, "-Infinity"),
            [((".zarray", "fill_value"), "non_finite_fill_value")],
        ),
        (
            _with_fill(COMPLEX_ZARRAY, [math.nan, math.inf]),
            _with_fill(COMPLEX_ZARRAY, ("NaN", "Infinity")),
            [
                ((".zarray", "fill_value", 0), "non_finite_fill_value"),
                ((".zarray", "fill_value", 1), "non_finite_fill_value"),
            ],
        ),
        (
            _with_fill(COMPLEX_ZARRAY, [1.5, -math.inf]),
            _with_fill(COMPLEX_ZARRAY, (1.5, "-Infinity")),
            [((".zarray", "fill_value", 1), "non_finite_fill_value")],
        ),
        # Not a known writer bug: left for the strict read to check.
        (_with_fill(NAN_ZARRAY, "NaN"), _with_fill(NAN_ZARRAY, "NaN"), []),
        (_with_fill(NAN_ZARRAY, 0.5), _with_fill(NAN_ZARRAY, 0.5), []),
        (
            _with_fill({**NAN_ZARRAY, "dtype": "<i4"}, math.nan),
            _with_fill({**NAN_ZARRAY, "dtype": "<i4"}, math.nan),
            [],
        ),
        (
            _with_fill({**NAN_ZARRAY, "zarr_format": 3}, math.nan),
            _with_fill({**NAN_ZARRAY, "zarr_format": 3}, math.nan),
            [],
        ),
        (
            _with_fill(COMPLEX_ZARRAY, [math.nan]),
            _with_fill(COMPLEX_ZARRAY, [math.nan]),
            [],
        ),
        ({".zgroup": {"zarr_format": 2}}, {".zgroup": {"zarr_format": 2}}, []),
        ({".zarray": 3}, {".zarray": 3}, []),
    ],
    ids=[
        "bare-nan",
        "bare-negative-infinity",
        "complex",
        "complex-one-component",
        "named",
        "finite",
        "int-dtype",
        "not-v2",
        "complex-of-another-length",
        "group",
        "not-a-document",
    ],
)
def test_repair_v2_node_writes_a_bare_non_finite_fill_value_by_name(
    files: dict[str, object], repaired: object, repairs: list[tuple[tuple[str | int, ...], str]]
) -> None:
    """zarr-python 3.0.0 to 3.0.6 wrote a non-finite fill value of a float or complex `.zarray` as a bare token, `NaN`, which is not JSON. Python's `json` module reads it as a float. `repair_node_metadata_v2` writes each one as the string the v2 spec uses, and reports its location. Anything else is left unchanged, and the input files are not modified."""
    before = copy.deepcopy(files)
    document, made = repair_node_metadata_v2(files)
    # `==` cannot compare NaN, which is never equal to itself, so compare the JSON text.
    assert json.dumps(document) == json.dumps(repaired)
    assert [(repair.loc, repair.kind) for repair in made] == repairs
    assert json.dumps(files) == json.dumps(before)
    if len(repairs) == 0:
        assert document is files


def test_read_repaired_v2_node_reads_what_the_strict_read_refuses() -> None:
    """The strict `read_node_metadata_v2` refuses the `.zarray` that zarr-python 3.0.0 to 3.0.6 wrote for a NaN fill value. `read_repaired_node_metadata_v2` reads the repaired files, with the same result as a strict read of them, and lists the repairs. A valid node reads the same either way."""
    files = {".zarray": json.loads(NAN_ZARRAY_BYTES)}
    assert [p.loc for p in read_node_metadata_v2(files).problems] == [("fill_value",)]
    repaired = read_repaired_node_metadata_v2(files)
    assert repaired.reading.problems == ()
    assert repaired.reading.metadata == ZarrV2ArrayMetadata(NAMED_ZARRAY)
    assert [r.kind for r in repaired.repairs] == ["non_finite_fill_value"]
    group = read_repaired_node_metadata_v2({".zgroup": {"zarr_format": 2}})
    assert group.reading.metadata == ZarrV2GroupMetadata({"zarr_format": 2})
    assert group.repairs == ()


def test_read_repaired_v2_node_reports_what_no_repair_applies_to() -> None:
    """A problem that no repair fixes is reported the same way the strict read reports it, alongside the repairs that were made."""
    files = {".zarray": {**NAN_ZARRAY, "order": "X"}}
    repaired = read_repaired_node_metadata_v2(files)
    assert [r.kind for r in repaired.repairs] == ["non_finite_fill_value"]
    assert [(p.loc, p.kind) for p in repaired.reading.problems] == [(("order",), "invalid_value")]
    assert repaired.reading.metadata is None


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
        # The root's .zgroup: no zarr-python version writes the member there.
        (
            {"zarr_consolidated_format": 1, "metadata": {".zgroup": ZGROUP_FROM_ZARR3}},
            {"zarr_consolidated_format": 1, "metadata": {".zgroup": ZGROUP_FROM_ZARR3}},
            [],
        ),
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
        (
            {"zarr_consolidated_format": 1, "metadata": {"a/.zarray": NAN_ZARRAY}},
            {"zarr_consolidated_format": 1, "metadata": {"a/.zarray": NAMED_ZARRAY}},
            [(("metadata", "a/.zarray", "fill_value"), "non_finite_fill_value")],
        ),
    ],
    ids=[
        "zarr-3-zgroup-entry",
        "clean",
        "root",
        "not-the-bug",
        "not-a-document",
        "bare-nan-in-zarray-entry",
    ],
)
def test_repair_v2_undoes_the_consolidated_metadata_zarr_3_writes_into_a_zgroup_entry(
    value: object, repaired: object, repairs: list[tuple[tuple[str | int, ...], str]]
) -> None:
    """zarr-python 3.x writes a `consolidated_metadata` member into each non-root `.zgroup` entry of a `.zmetadata`, but the v2 group document does not allow that member. `repair_consolidated_metadata_v2` removes it and reports its location. Anything else is left unchanged, and the input document is not modified."""
    before = copy.deepcopy(value)
    document, made = repair_consolidated_metadata_v2(value)
    assert document == repaired
    assert [(repair.loc, repair.kind) for repair in made] == repairs
    assert value == before
    if len(repairs) == 0:
        assert document is value


def test_read_repaired_consolidated_v2_reads_what_the_strict_read_refuses() -> None:
    """The strict `ZarrV2ConsolidatedMetadata` refuses the `.zmetadata` that zarr-python 3.x writes, reporting the problem at the entry's member. `read_repaired_consolidated_metadata_v2` reads the repaired document, holds its model, and lists the repairs."""
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


def test_read_repaired_consolidated_v2_writes_a_bare_non_finite_fill_value_by_name() -> None:
    """A `.zarray` entry of a `.zmetadata` with a bare NaN fill value, as zarr-python 3.0.0 to 3.0.6 wrote it, is refused by the strict read and read by `read_repaired_consolidated_metadata_v2`, after the same repair as for a `.zarray` file."""
    value = {
        **ZMETADATA_CLEAN,
        "metadata": {**ZMETADATA_CLEAN["metadata"], "b/a/.zarray": NAN_ZARRAY},
    }
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ConsolidatedMetadata(value)
    assert [p.loc for p in raised.value.problems] == [("metadata", "b/a/.zarray", "fill_value")]
    repaired = read_repaired_consolidated_metadata_v2(value)
    assert repaired.problems == ()
    assert repaired.metadata is not None
    assert repaired.metadata.metadata["b/a"] == ZarrV2ArrayMetadata(NAMED_ZARRAY)
    assert [(r.loc, r.kind) for r in repaired.repairs] == [
        (("metadata", "b/a/.zarray", "fill_value"), "non_finite_fill_value")
    ]
