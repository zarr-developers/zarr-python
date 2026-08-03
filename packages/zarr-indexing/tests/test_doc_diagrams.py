from __future__ import annotations

import re
from pathlib import Path
from typing import NotRequired, TypedDict, Unpack, cast, get_args
from xml.etree import ElementTree

import pytest

import docs.diagrams.render as diagram_render
import docs.diagrams.specifications as diagram_specs
from docs.diagrams.render import STYLESHEET_NAME, render_all, render_figure
from docs.diagrams.specifications import (
    FIGURES,
    ArrowSpec,
    AxisSpec,
    ElementSpec,
    FigureSpec,
    GridSpec,
    NodeSpec,
    SemanticRole,
    TextSpec,
    validate_figure,
)

DOCS = Path(__file__).parents[1] / "docs"
PALETTES = diagram_specs.PALETTES
ROLE_CUES = diagram_specs.ROLE_CUES


class WriteTextOptions(TypedDict):
    encoding: NotRequired[str | None]
    errors: NotRequired[str | None]
    newline: NotRequired[str | None]


def figure(figure_id: str, *, elements: tuple[ElementSpec, ...]) -> FigureSpec:
    return {
        "id": figure_id,
        "title": f"Figure {figure_id}",
        "description": "A deterministic renderer test figure.",
        "width": 240,
        "height": 120,
        "elements": elements,
    }


def malformed_figure(figure_id: str, *, elements: tuple[object, ...]) -> FigureSpec:
    """Construct an intentionally invalid runtime input for validation tests."""
    return cast(
        FigureSpec,
        {
            "id": figure_id,
            "title": f"Figure {figure_id}",
            "description": "A deterministic renderer test figure.",
            "width": 240,
            "height": 120,
            "elements": elements,
        },
    )


GRID: GridSpec = {
    "id": "grid",
    "kind": "grid",
    "role": "selected",
    "rows": 2,
    "columns": 3,
    "cell_size": 20,
    "origin": (10, 10),
    "selected": ((1, 2),),
    "chunk_shape": (1, 2),
}
AXIS: AxisSpec = {
    "id": "axis",
    "kind": "axis",
    "role": "text",
    "x": 10,
    "y": 90,
    "length": 100,
    "orientation": "horizontal",
    "start": 0,
    "stop": 5,
}
LEFT_NODE: NodeSpec = {
    "id": "left",
    "kind": "node",
    "role": "source",
    "x": 10,
    "y": 10,
    "width": 40,
    "height": 24,
    "label": "source",
}
RIGHT_NODE: NodeSpec = {
    "id": "right",
    "kind": "node",
    "role": "request",
    "x": 100,
    "y": 10,
    "width": 40,
    "height": 24,
    "label": "request",
}
ARROW: ArrowSpec = {
    "id": "arrow",
    "kind": "arrow",
    "role": "text",
    "source": "left",
    "target": "right",
    "label": "maps to",
}
TEXT: TextSpec = {
    "id": "text",
    "kind": "text",
    "role": "cell-domain",
    "x": 10,
    "y": 110,
    "label": "cell domain",
}
VALID_ELEMENT_COMBINATIONS: tuple[tuple[ElementSpec, ...], ...] = (
    (GRID,),
    (AXIS,),
    (LEFT_NODE,),
    (LEFT_NODE, RIGHT_NODE, ARROW),
    (TEXT,),
)


@pytest.mark.parametrize("elements", VALID_ELEMENT_COMBINATIONS)
def test_render_figure_accepts_reasonable_element_combinations(
    elements: tuple[ElementSpec, ...],
) -> None:
    spec = figure("valid", elements=elements)
    first = render_figure(spec)
    second = render_figure(spec)
    assert first == second
    root = ElementTree.fromstring(first)
    assert root.tag.endswith("svg")
    assert root.attrib["role"] == "img"


def test_grid_size_must_be_positive() -> None:
    invalid_grid = {**GRID, "rows": 0}
    with pytest.raises(ValueError, match="invalid-grid.*rows"):
        validate_figure(malformed_figure("invalid-grid", elements=(invalid_grid,)))


def test_selected_coordinate_must_be_inside_grid() -> None:
    invalid_grid = {**GRID, "selected": ((2, 0),)}
    with pytest.raises(ValueError, match="outside-grid.*selected"):
        validate_figure(malformed_figure("outside-grid", elements=(invalid_grid,)))


def test_grid_show_coordinates_must_be_boolean() -> None:
    invalid_grid = {**GRID, "show_coordinates": "yes"}
    with pytest.raises(ValueError, match="show-coordinates.*show_coordinates.*boolean"):
        validate_figure(malformed_figure("show-coordinates", elements=(invalid_grid,)))


def test_grid_show_selected_coordinates_must_be_boolean() -> None:
    invalid_grid = {**GRID, "show_selected_coordinates": "yes"}
    with pytest.raises(
        ValueError, match="show-selected-coordinates.*show_selected_coordinates.*boolean"
    ):
        validate_figure(malformed_figure("show-selected-coordinates", elements=(invalid_grid,)))


def test_grid_show_value_prefix_must_be_boolean() -> None:
    invalid_grid = {**GRID, "show_value_prefix": "yes"}
    with pytest.raises(ValueError, match="show-value-prefix.*show_value_prefix.*boolean"):
        validate_figure(malformed_figure("show-value-prefix", elements=(invalid_grid,)))


def test_text_boxed_must_be_boolean() -> None:
    invalid_text = {**TEXT, "boxed": "yes"}
    with pytest.raises(ValueError, match="boxed-text.*boxed.*boolean"):
        validate_figure(malformed_figure("boxed-text", elements=(invalid_text,)))


def test_text_boxed_controls_the_visible_label_box() -> None:
    boxed: TextSpec = {**TEXT, "id": "boxed"}
    unboxed: TextSpec = {**TEXT, "id": "unboxed", "boxed": False}
    root = ElementTree.fromstring(render_figure(figure("text-boxes", elements=(boxed, unboxed))))
    groups = {
        element.attrib["id"]: element
        for element in root.iter()
        if element.attrib.get("id") in {"text-boxes-boxed", "text-boxes-unboxed"}
    }

    assert any(child.tag.endswith("rect") for child in groups["text-boxes-boxed"])
    assert not any(child.tag.endswith("rect") for child in groups["text-boxes-unboxed"])
    assert any(child.tag.endswith("text") for child in groups["text-boxes-unboxed"])


def test_figure_show_legend_must_be_boolean() -> None:
    spec = figure("show-legend", elements=(TEXT,))
    spec["show_legend"] = "yes"  # type: ignore[typeddict-item]
    with pytest.raises(ValueError, match="show-legend.*show_legend.*boolean"):
        validate_figure(spec)


def test_element_ids_must_be_unique() -> None:
    duplicate = {**TEXT, "id": "grid"}
    with pytest.raises(ValueError, match="duplicate-id.*grid"):
        validate_figure(malformed_figure("duplicate-id", elements=(GRID, duplicate)))


def test_title_must_not_be_empty() -> None:
    spec = figure("empty-title", elements=(TEXT,))
    spec["title"] = ""
    with pytest.raises(ValueError, match="empty-title.*title"):
        validate_figure(spec)


def test_description_must_not_be_empty() -> None:
    spec = figure("empty-description", elements=(TEXT,))
    spec["description"] = ""
    with pytest.raises(ValueError, match="empty-description.*description"):
        validate_figure(spec)


def test_semantic_role_must_be_known() -> None:
    invalid_text = {**TEXT, "role": "unknown"}
    with pytest.raises(ValueError, match="unknown-role.*role"):
        validate_figure(malformed_figure("unknown-role", elements=(invalid_text,)))


def test_unhashable_semantic_role_has_a_scoped_validation_error() -> None:
    invalid_text = {**TEXT, "role": list[object]()}
    with pytest.raises(ValueError) as error:
        validate_figure(malformed_figure("unhashable-role", elements=(invalid_text,)))
    assert str(error.value) == "unhashable-role: text.role must be a known semantic role"


def test_unhashable_element_kind_has_a_scoped_validation_error() -> None:
    invalid_text = {**TEXT, "kind": dict[object, object]()}
    with pytest.raises(ValueError) as error:
        validate_figure(malformed_figure("unhashable-kind", elements=(invalid_text,)))
    assert str(error.value) == "unhashable-kind: text.kind must be a known element kind"


def test_figure_id_must_be_a_safe_xml_fragment_token() -> None:
    spec = figure("two figure ids", elements=(TEXT,))
    with pytest.raises(ValueError) as error:
        validate_figure(spec)
    assert str(error.value) == ("two figure ids: id must match [A-Za-z][A-Za-z0-9_-]*")


def test_element_id_must_be_a_safe_xml_fragment_token() -> None:
    invalid_text = {**TEXT, "id": "two element ids"}
    with pytest.raises(ValueError) as error:
        validate_figure(malformed_figure("unsafe-element-id", elements=(invalid_text,)))
    assert str(error.value) == ("unsafe-element-id: elements.id must match [A-Za-z][A-Za-z0-9_-]*")


def test_arrow_target_must_resolve() -> None:
    broken_arrow: ArrowSpec = {
        "id": "a",
        "kind": "arrow",
        "role": "text",
        "source": "left",
        "target": "missing",
        "label": "broken",
    }
    spec = figure("broken-arrow", elements=(LEFT_NODE, broken_arrow))
    with pytest.raises(ValueError, match="broken-arrow.*missing"):
        validate_figure(spec)


def test_grid_chunk_shape_renders_complete_chunk_outlines() -> None:
    grid: GridSpec = {
        **GRID,
        "rows": 5,
        "columns": 7,
        "cell_size": 10,
        "origin": (5, 7),
        "chunk_shape": (3, 4),
    }
    root = ElementTree.fromstring(render_figure(figure("chunked", elements=(grid,))))
    outlines = [
        element.attrib
        for element in root.iter()
        if element.tag.endswith("rect") and element.attrib.get("class") == "zi-chunk-boundary"
    ]
    assert outlines == [
        {"class": "zi-chunk-boundary", "height": "30", "width": "40", "x": "5", "y": "7"},
        {"class": "zi-chunk-boundary", "height": "30", "width": "30", "x": "45", "y": "7"},
        {"class": "zi-chunk-boundary", "height": "20", "width": "40", "x": "5", "y": "37"},
        {"class": "zi-chunk-boundary", "height": "20", "width": "30", "x": "45", "y": "37"},
    ]
    unchunked: GridSpec = {**grid, "chunk_shape": None}
    unchunked_root = ElementTree.fromstring(
        render_figure(figure("unchunked", elements=(unchunked,)))
    )
    assert not any(
        element.attrib.get("class") == "zi-chunk-boundary" for element in unchunked_root.iter()
    )


def test_grid_show_coordinates_labels_only_unselected_cells() -> None:
    grid: GridSpec = {**GRID, "show_coordinates": True}
    root = ElementTree.fromstring(render_figure(figure("coordinates", elements=(grid,))))
    labels = [
        element
        for element in root.iter()
        if element.attrib.get("class") == "zi-unselected-coordinate-label"
    ]

    assert [label.text for label in labels] == ["(0, 0)", "(0, 1)", "(0, 2)", "(1, 0)", "(1, 1)"]
    assert all(label.attrib["aria-hidden"] == "true" for label in labels)


def test_grid_values_render_coordinates_and_values_for_every_cell() -> None:
    grid = cast(
        GridSpec,
        {
            **GRID,
            "values": ((0, 1, 2), (3, 4, 5)),
            "show_coordinates": True,
        },
    )
    root = ElementTree.fromstring(render_figure(figure("grid-values", elements=(grid,))))
    coordinate_labels = [
        element.text
        for element in root.iter()
        if "zi-cell-coordinate" in element.attrib.get("class", "").split()
    ]
    value_labels = [
        element.text
        for element in root.iter()
        if "zi-cell-value" in element.attrib.get("class", "").split()
    ]

    assert coordinate_labels == [
        "coord: (0, 0)",
        "coord: (0, 1)",
        "coord: (0, 2)",
        "coord: (1, 0)",
        "coord: (1, 1)",
        "coord: (1, 2)",
    ]
    assert value_labels == [
        "value: 0",
        "value: 1",
        "value: 2",
        "value: 3",
        "value: 4",
        "value: 5",
    ]


def test_grid_values_must_have_one_row_per_grid_row() -> None:
    invalid_grid = {**GRID, "values": ((0, 1, 2),)}
    with pytest.raises(ValueError, match=r"grid-values-rows.*grid\.values.*2 rows"):
        validate_figure(malformed_figure("grid-values-rows", elements=(invalid_grid,)))


def test_grid_values_must_have_one_value_per_grid_column() -> None:
    invalid_grid = {**GRID, "values": ((0, 1), (2, 3, 4))}
    with pytest.raises(ValueError, match=r"grid-values-columns.*grid\.values.*3 columns"):
        validate_figure(malformed_figure("grid-values-columns", elements=(invalid_grid,)))


def test_sequence_renders_coordinate_value_cells_and_can_anchor_an_arrow() -> None:
    sequence = cast(
        ElementSpec,
        {
            "id": "request-sequence",
            "kind": "sequence",
            "role": "request",
            "x": 10,
            "y": 10,
            "width": 90,
            "height": 72,
            "orientation": "horizontal",
            "coordinates": (0, 1, 2),
            "values": (4, 5, 6),
        },
    )
    arrow = {**ARROW, "source": "request-sequence"}
    root = ElementTree.fromstring(
        render_figure(malformed_figure("request-sequence", elements=(sequence, RIGHT_NODE, arrow)))
    )
    sequence_group = next(
        element
        for element in root.iter()
        if element.attrib.get("id") == "request-sequence-request-sequence"
    )

    assert [
        element.text
        for element in sequence_group.iter()
        if "zi-cell-coordinate" in element.attrib.get("class", "").split()
    ] == ["coord: 0", "coord: 1", "coord: 2"]
    assert [
        element.text
        for element in sequence_group.iter()
        if "zi-cell-value" in element.attrib.get("class", "").split()
    ] == ["value: 4", "value: 5", "value: 6"]
    assert any(element.attrib.get("class") == "zi-arrow-line" for element in root.iter())
    assert [
        {key: element.attrib[key] for key in ("height", "width", "x", "y")}
        for element in sequence_group
        if element.attrib.get("class") == "zi-sequence-cell"
    ] == [
        {"height": "72", "width": "30.0", "x": "10.0", "y": "10"},
        {"height": "72", "width": "30.0", "x": "40.0", "y": "10"},
        {"height": "72", "width": "30.0", "x": "70.0", "y": "10"},
    ]


def test_sequence_coordinates_and_values_must_have_equal_lengths() -> None:
    invalid_sequence = {
        "id": "sequence",
        "kind": "sequence",
        "role": "request",
        "x": 10,
        "y": 10,
        "width": 90,
        "height": 72,
        "orientation": "vertical",
        "coordinates": (0, 1),
        "values": (4,),
    }
    with pytest.raises(ValueError, match=r"sequence-length.*sequence\.values.*coordinates"):
        validate_figure(malformed_figure("sequence-length", elements=(invalid_sequence,)))


def test_sequence_orientation_must_be_horizontal_or_vertical() -> None:
    invalid_sequence = {
        "id": "sequence",
        "kind": "sequence",
        "role": "request",
        "x": 10,
        "y": 10,
        "width": 90,
        "height": 72,
        "orientation": "diagonal",
        "coordinates": (0, 1),
        "values": (4, 5),
    }
    with pytest.raises(ValueError, match=r"sequence-orientation.*sequence\.orientation"):
        validate_figure(malformed_figure("sequence-orientation", elements=(invalid_sequence,)))


def test_arrow_label_must_be_a_string() -> None:
    arrow_without_label = {
        "id": "arrow",
        "kind": "arrow",
        "role": "text",
        "source": "left",
        "target": "right",
    }
    spec = malformed_figure(
        "missing-arrow-label", elements=(LEFT_NODE, RIGHT_NODE, arrow_without_label)
    )
    with pytest.raises(ValueError, match="missing-arrow-label.*arrow.label"):
        validate_figure(spec)


@pytest.mark.parametrize("label_offset", [(1,), None, (0, "up"), (False, 1)])
def test_arrow_label_offset_must_be_a_pair_of_integers(label_offset: object) -> None:
    arrow_with_invalid_offset = {**ARROW, "label_offset": label_offset}
    spec = malformed_figure(
        "invalid-arrow-offset", elements=(LEFT_NODE, RIGHT_NODE, arrow_with_invalid_offset)
    )
    with pytest.raises(ValueError, match=r"invalid-arrow-offset.*arrow\.label_offset"):
        validate_figure(spec)


def test_grid_geometry_must_not_be_boolean() -> None:
    grid_with_boolean_rows = {**GRID, "rows": True}
    with pytest.raises(ValueError, match="boolean-grid.*grid.rows"):
        validate_figure(malformed_figure("boolean-grid", elements=(grid_with_boolean_rows,)))


def test_figure_geometry_must_not_be_boolean() -> None:
    spec = cast(
        FigureSpec,
        {
            **figure("boolean-figure", elements=(TEXT,)),
            "width": True,
        },
    )
    with pytest.raises(ValueError, match="boolean-figure.*width"):
        validate_figure(spec)


def test_legend_bottom_padding_must_not_be_negative() -> None:
    spec = cast(
        FigureSpec,
        {
            **figure("negative-legend-padding", elements=(TEXT,)),
            "legend_bottom_padding": -1,
        },
    )
    with pytest.raises(ValueError, match="negative-legend-padding.*legend_bottom_padding"):
        validate_figure(spec)


def test_legend_bottom_padding_must_be_an_integer() -> None:
    spec = cast(
        FigureSpec,
        {
            **figure("boolean-legend-padding", elements=(TEXT,)),
            "legend_bottom_padding": True,
        },
    )
    with pytest.raises(ValueError, match="boolean-legend-padding.*legend_bottom_padding"):
        validate_figure(spec)


@pytest.fixture
def generated_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(diagram_render, "FIGURES", (figure("generated", elements=(TEXT,)),))
    render_all(tmp_path, check=False)
    return tmp_path


def _snapshot(output_dir: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(output_dir)): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }


def test_render_all_update_then_check_writes_deterministic_utf8_lf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagram_render, "FIGURES", (figure("generated", elements=(TEXT,)),))
    original_write_text = Path.write_text
    write_options: list[WriteTextOptions] = []

    def write_text_with_options(self: Path, data: str, **kwargs: Unpack[WriteTextOptions]) -> int:
        write_options.append(kwargs)
        return original_write_text(self, data, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text_with_options)
    render_all(tmp_path, check=False)
    render_all(tmp_path, check=True)
    assert set(_snapshot(tmp_path)) == {"diagrams/generated.svg", STYLESHEET_NAME}
    assert write_options == [{"encoding": "utf-8", "newline": "\n"}] * 2


@pytest.mark.parametrize(
    ("filename", "replacement"),
    [
        ("generated.svg", b"changed"),
        (STYLESHEET_NAME, b"changed stylesheet"),
        ("missing.svg", None),
        ("orphan.svg", b"orphan"),
    ],
)
def test_render_all_check_reports_the_exact_stale_path(
    generated_assets: Path, filename: str, replacement: bytes | None
) -> None:
    relative_path = (
        Path("diagrams/generated.svg")
        if filename == "missing.svg"
        else Path(filename)
        if filename == STYLESHEET_NAME
        else Path("diagrams") / filename
    )
    path = generated_assets / relative_path
    if replacement is None:
        path.unlink()
    else:
        path.write_bytes(replacement)
    before = _snapshot(generated_assets)
    with pytest.raises(RuntimeError) as error:
        render_all(generated_assets, check=True)
    assert str(error.value) == f"stale generated diagram: {path}"
    assert _snapshot(generated_assets) == before


def test_render_all_update_prunes_orphaned_svgs_then_passes_check(
    generated_assets: Path,
) -> None:
    orphan = generated_assets / "diagrams" / "orphan.svg"
    nested_orphan = generated_assets / "diagrams" / "retired" / "old.svg"
    orphan.write_text("orphan", encoding="utf-8")
    nested_orphan.parent.mkdir()
    nested_orphan.write_text("old", encoding="utf-8")

    render_all(generated_assets, check=False)

    assert not orphan.exists()
    assert not nested_orphan.exists()
    assert nested_orphan.parent.is_dir()
    render_all(generated_assets, check=True)


def test_render_all_update_preserves_unrelated_files_and_directories(
    generated_assets: Path,
) -> None:
    unrelated_svg = generated_assets / "manual.svg"
    unrelated_text = generated_assets / "diagrams" / "notes.txt"
    unrelated_directory = generated_assets / "diagrams" / "manual-assets"
    unrelated_nested_file = unrelated_directory / "notes.txt"
    unrelated_svg.write_text("manual", encoding="utf-8")
    unrelated_text.write_text("notes", encoding="utf-8")
    unrelated_directory.mkdir()
    unrelated_nested_file.write_text("nested notes", encoding="utf-8")
    before = {
        path: path.read_bytes() for path in (unrelated_svg, unrelated_text, unrelated_nested_file)
    }

    render_all(generated_assets, check=False)

    assert unrelated_directory.is_dir()
    assert {path: path.read_bytes() for path in before} == before


def test_render_all_check_includes_stylesheet_and_does_not_mutate(generated_assets: Path) -> None:
    stylesheet = generated_assets / STYLESHEET_NAME
    assert stylesheet.exists()
    before = _snapshot(generated_assets)
    render_all(generated_assets, check=True)
    assert _snapshot(generated_assets) == before


def _linear_channel(value: int) -> float:
    channel = value / 255
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio for two six-digit hexadecimal colours."""
    luminances: list[float] = []
    for color in (foreground, background):
        red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
        luminances.append(
            0.2126 * _linear_channel(red)
            + 0.7152 * _linear_channel(green)
            + 0.0722 * _linear_channel(blue)
        )
    lighter, darker = sorted(luminances, reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_semantic_palettes_are_accessible() -> None:
    for scheme in ("default", "slate"):
        assert set(PALETTES[scheme]) == set(get_args(SemanticRole))
        for role, colors in PALETTES[scheme].items():
            assert contrast_ratio(colors["text"], colors["fill"]) >= 4.5, role
            assert ROLE_CUES[role]


def test_figure_registry_has_the_approved_accessible_conclusions() -> None:
    assert {figure_spec["id"]: figure_spec["description"] for figure_spec in FIGURES} == {
        "indexing-selection": (
            "source[2:5] maps source coordinates 2, 3, 4, containing values 12, 13, 14, "
            "to result coordinates 0, 1, 2, respectively."
        ),
        "coordinate-addresses": "domain [-2, 3) gives equal status to -2, -1, 0, 1, 2.",
        "prepend-chunk": "[0, 6) becomes [-3, 6) while old coordinates stay fixed.",
        "transform-mapping": "request i maps to source i + 2.",
        "compose-views": "two view maps collapse into one direct map.",
        "chunk-overlay": (
            "touched global chunk domains map selected cells to zero-origin "
            "chunk-local coordinates."
        ),
        "projection-pair": "one cell domain points to request and chunk-local spaces.",
        "orthogonal-contrast": "rows [4, 1, 1] retain order and duplication.",
        "system-memory-chunk-lifecycle": (
            "chunks move through queued and loading states before system-memory residency, "
            "with explicit failure, retry, eviction, and reload paths."
        ),
    }


def test_system_memory_lifecycle_figure_has_the_documented_transitions() -> None:
    spec = next(item for item in FIGURES if item["id"] == "system-memory-chunk-lifecycle")
    nodes = {
        element["id"]: element["label"] for element in spec["elements"] if element["kind"] == "node"
    }
    transitions = {
        (element["source"], element["target"], element["label"])
        for element in spec["elements"]
        if element["kind"] == "arrow"
    }

    assert nodes == {
        "new": "NEW",
        "queued": "QUEUED",
        "loading": "LOADING",
        "ready": "READY: system memory",
        "failed": "FAILED",
        "evicted": "EVICTED",
    }
    assert transitions == {
        ("new", "queued", "request"),
        ("queued", "loading", "drain"),
        ("loading", "ready", "read succeeds"),
        ("loading", "failed", "read fails"),
        ("failed", "queued", "explicit retry"),
        ("ready", "evicted", "LRU pressure"),
        ("evicted", "queued", "request again"),
    }


def test_system_memory_lifecycle_legend_reserves_24_units_below_visible_content() -> None:
    spec = next(item for item in FIGURES if item["id"] == "system-memory-chunk-lifecycle")
    assert spec.get("legend_bottom_padding") == 24

    root = ElementTree.fromstring(render_figure(spec))
    legend = next(
        element
        for element in root.iter()
        if element.attrib.get("id") == "system-memory-chunk-lifecycle-legend"
    )
    swatches = [
        element
        for element in legend.iter()
        if "zi-cue-swatch" in element.attrib.get("class", "").split()
    ]
    labels = [
        element
        for element in legend.iter()
        if "zi-role-label" in element.attrib.get("class", "").split()
    ]
    content_bottom = spec["height"] - 24
    role_label_font_size = 12

    assert swatches
    assert labels
    assert all(
        float(element.attrib["y"]) + float(element.attrib["height"]) <= content_bottom
        for element in swatches
    )
    assert all(
        float(element.attrib["y"]) + role_label_font_size / 2 <= content_bottom
        for element in labels
    )


def test_default_legend_bottom_padding_preserves_existing_coordinates() -> None:
    root = ElementTree.fromstring(render_figure(figure("default-legend", elements=(TEXT,))))
    swatch = next(
        element
        for element in root.iter()
        if "zi-cue-swatch" in element.attrib.get("class", "").split()
    )
    assert (swatch.attrib["y"], swatch.attrib["height"]) == ("88", "20")


def _grid_element(spec: FigureSpec, element_id: str) -> GridSpec:
    for element in spec["elements"]:
        if element["id"] == element_id and element["kind"] == "grid":
            return element
    raise AssertionError(f"missing grid {element_id!r}")


def test_chunk_overlay_uses_the_canonical_two_dimensional_source_data() -> None:
    chunk_overlay = next(item for item in FIGURES if item["id"] == "chunk-overlay")
    grid = _grid_element(chunk_overlay, "chunked-source")

    assert grid["kind"] == "grid"
    assert (grid["rows"], grid["columns"]) == (3, 4)
    assert grid["selected"] == ((1, 0), (1, 1), (1, 2), (1, 3))
    assert grid.get("values") == ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
    assert grid.get("chunk_shape") == (2, 2)


def test_chunk_overlay_svg_has_12_cells_and_complete_2_by_2_chunk_outlines() -> None:
    chunk_overlay = next(item for item in FIGURES if item["id"] == "chunk-overlay")
    overlay_root = ElementTree.fromstring(render_figure(chunk_overlay))
    overlay_grid = next(
        element
        for element in overlay_root.iter()
        if element.attrib.get("id") == "chunk-overlay-chunked-source"
    )
    cells = [
        element
        for element in overlay_grid.iter()
        if element.tag.endswith("rect")
        and "zi-grid-cell" in element.attrib.get("class", "").split()
    ]
    assert len(cells) == 12
    outlines = [
        element.attrib
        for element in overlay_grid
        if element.tag.endswith("rect") and element.attrib.get("class") == "zi-chunk-boundary"
    ]
    assert outlines == [
        {
            "class": "zi-chunk-boundary",
            "height": "144",
            "width": "144",
            "x": "76",
            "y": "140",
        },
        {
            "class": "zi-chunk-boundary",
            "height": "144",
            "width": "144",
            "x": "220",
            "y": "140",
        },
        {
            "class": "zi-chunk-boundary",
            "height": "72",
            "width": "144",
            "x": "76",
            "y": "284",
        },
        {
            "class": "zi-chunk-boundary",
            "height": "72",
            "width": "144",
            "x": "220",
            "y": "284",
        },
    ]


def test_chunk_overlay_shows_faded_coordinates_for_unselected_cells() -> None:
    figure_id = "chunk-overlay"
    grid_id = "chunked-source"
    spec = next(figure_spec for figure_spec in FIGURES if figure_spec["id"] == figure_id)
    root = ElementTree.fromstring(render_figure(spec))
    grid = next(
        element for element in root.iter() if element.attrib.get("id") == f"{figure_id}-{grid_id}"
    )
    unselected_labels = [
        element
        for element in grid.iter()
        if "zi-unselected-coordinate-label" in element.attrib.get("class", "").split()
    ]
    selected_labels = [
        element
        for element in grid.iter()
        if "zi-selected-coordinate-label" in element.attrib.get("class", "").split()
    ]

    assert len(unselected_labels) == 8
    assert len(selected_labels) == 4
    assert {label.text for label in unselected_labels}.isdisjoint(
        {label.text for label in selected_labels}
    )


def test_indexing_selection_shows_one_dimensional_result_to_source_correspondence() -> None:
    spec = next(item for item in FIGURES if item["id"] == "indexing-selection")
    assert spec["description"] == (
        "source[2:5] maps source coordinates 2, 3, 4, containing values 12, 13, 14, "
        "to result coordinates 0, 1, 2, respectively."
    )

    source = _grid_element(spec, "source-values")
    assert (source["rows"], source["columns"]) == (1, 6)
    assert source["selected"] == ((0, 2), (0, 3), (0, 4))
    assert source.get("values") == ((10, 11, 12, 13, 14, 15),)

    text = {element["id"]: element for element in spec["elements"] if element["kind"] == "text"}
    values = {
        int(element["id"].removeprefix("result-value-")): element
        for element in spec["elements"]
        if element["kind"] == "node" and element["id"].startswith("result-value-")
    }
    for result_coordinate, source_coordinate in enumerate((2, 3, 4)):
        center = values[result_coordinate]["x"] + values[result_coordinate]["width"] // 2
        assert text[f"result-coordinate-{result_coordinate}"]["label"] == str(result_coordinate)
        assert text[f"source-coordinate-{result_coordinate}"]["label"] == str(source_coordinate)
        assert text[f"result-coordinate-{result_coordinate}"]["x"] == center
        assert text[f"source-coordinate-{result_coordinate}"]["x"] == center
        assert values[result_coordinate]["label"] == str(source_coordinate + 10)

    visible = " ".join(
        element["label"] for element in spec["elements"] if element["kind"] == "text"
    ).lower()
    assert "axis" not in visible
    assert "shape" not in visible
    assert all(element["kind"] != "arrow" for element in spec["elements"])
    assert spec.get("show_legend") is False


def test_transform_mapping_shows_the_slice_offset() -> None:
    spec = next(item for item in FIGURES if item["id"] == "transform-mapping")
    nodes = {
        element["id"]: element["label"] for element in spec["elements"] if element["kind"] == "node"
    }

    assert spec["description"] == "request i maps to source i + 2."
    assert nodes == {"request-i": "request i", "source-coordinate": "source i + 2"}


def test_chunk_overlay_uses_a_phone_readable_stacked_layout() -> None:
    """The 390 px audit viewport must keep key labels at 12 CSS px without scrolling."""
    overlay = next(figure_spec for figure_spec in FIGURES if figure_spec["id"] == "chunk-overlay")
    grid = _grid_element(overlay, "chunked-source")
    root = ElementTree.fromstring(render_figure(overlay))
    rendered_phone_width = 386
    svg_label_size = 14

    assert svg_label_size * rendered_phone_width / overlay["width"] >= 12
    assert overlay["height"] > overlay["width"]
    grid_left, grid_top = grid["origin"]
    grid_right = grid_left + grid["columns"] * grid["cell_size"]
    grid_bottom = grid_top + grid["rows"] * grid["cell_size"]
    assert grid_left == overlay["width"] - grid_right

    annotation_boxes = {
        element.attrib["id"].removeprefix("chunk-overlay-"): next(
            child for child in element if child.tag.endswith("rect")
        )
        for element in root.iter()
        if element.attrib.get("id", "").removeprefix("chunk-overlay-")
        in {
            "chunk-zero-coords",
            "chunk-zero-domain",
            "chunk-zero-local",
            "chunk-one-coords",
            "chunk-one-domain",
            "chunk-one-local",
            "selection-label",
        }
    }
    for element_id in ("chunk-zero-coords", "chunk-zero-domain", "chunk-zero-local"):
        box = annotation_boxes[element_id]
        assert float(box.attrib["y"]) + float(box.attrib["height"]) <= grid_top
    for element_id in (
        "chunk-one-coords",
        "chunk-one-domain",
        "chunk-one-local",
        "selection-label",
    ):
        assert float(annotation_boxes[element_id].attrib["y"]) >= grid_bottom

    chunk_overlay_rule = diagram_render.STYLESHEET.split(
        '.zi-figure[aria-labelledby~="chunk-overlay-title"] {', maxsplit=1
    )[1].split("}", maxsplit=1)[0]
    assert "max-width: 528px;" in chunk_overlay_rule
    assert "overflow" not in chunk_overlay_rule


def test_chunk_overlay_distinguishes_global_chunk_metadata_from_local_outputs() -> None:
    """Global chunk labels must not borrow the chunk-local role or cue."""
    overlay = next(figure_spec for figure_spec in FIGURES if figure_spec["id"] == "chunk-overlay")
    annotations = {
        element["id"]: (element["role"], element["label"])
        for element in overlay["elements"]
        if element["kind"] == "text"
    }

    assert annotations["chunk-zero-coords"] == ("source", "global chunk_coords (0, 0)")
    assert annotations["chunk-zero-domain"] == (
        "source",
        "global chunk_domain [0, 2) \N{MULTIPLICATION SIGN} [0, 2)",
    )
    assert annotations["chunk-zero-local"] == (
        "chunk-local",
        "chunk-local selected (1, 0), (1, 1)",
    )
    assert annotations["chunk-one-coords"] == ("source", "global chunk_coords (0, 1)")
    assert annotations["chunk-one-domain"] == (
        "source",
        "global chunk_domain [0, 2) \N{MULTIPLICATION SIGN} [2, 4)",
    )
    assert annotations["chunk-one-local"] == (
        "chunk-local",
        "chunk-local selected (1, 0), (1, 1)",
    )


def test_chunk_overlay_selected_cells_have_visible_coordinate_cues() -> None:
    """Each selected cell must carry the coordinate label promised by ROLE_CUES."""
    overlay = next(figure_spec for figure_spec in FIGURES if figure_spec["id"] == "chunk-overlay")
    root = ElementTree.fromstring(render_figure(overlay))
    selected_groups = [
        element
        for element in root.iter()
        if "zi-selected-cell" in element.attrib.get("class", "").split()
    ]

    assert [element.attrib["aria-label"] for element in selected_groups] == [
        "selected coordinate (1, 0), value 4",
        "selected coordinate (1, 1), value 5",
        "selected coordinate (1, 2), value 6",
        "selected coordinate (1, 3), value 7",
    ]
    assert all(element.attrib["data-semantic-role"] == "selected" for element in selected_groups)
    assert all(
        element.attrib["data-non-color-cue"] == ROLE_CUES["selected"] for element in selected_groups
    )
    assert [
        child.text
        for element in selected_groups
        for child in element
        if "zi-selected-coordinate-label" in child.attrib.get("class", "").split()
    ] == ["coord: (1, 0)", "coord: (1, 1)", "coord: (1, 2)", "coord: (1, 3)"]


def test_chunk_boundaries_are_solid_while_chunk_local_cues_remain_dashed() -> None:
    chunk_boundary_rule = diagram_render.STYLESHEET.split(
        ".zi-figure .zi-chunk-boundary {", maxsplit=1
    )[1].split("}", maxsplit=1)[0]
    source_chunk_boundary_rule = diagram_render.STYLESHEET.split(
        ".zi-figure .zi-role-source > rect.zi-chunk-boundary {", maxsplit=1
    )[1].split("}", maxsplit=1)[0]
    chunk_local_rule = diagram_render.STYLESHEET.split(
        ".zi-figure .zi-role-chunk-local > rect", maxsplit=1
    )[1].split("}", maxsplit=1)[0]

    assert "stroke-width: 4;" in chunk_boundary_rule
    assert "stroke-width: 4;" in source_chunk_boundary_rule
    assert "stroke-dasharray" not in chunk_boundary_rule
    assert "stroke-dasharray: 7 4;" in chunk_local_rule


def test_unselected_coordinate_labels_use_the_faded_source_color() -> None:
    coordinate_rule = diagram_render.STYLESHEET.split(
        ".zi-figure .zi-role-source text.zi-unselected-coordinate-label {", maxsplit=1
    )[1].split("}", maxsplit=1)[0]

    assert "fill: var(--zi-source-stroke" in coordinate_rule
    assert "font-size: 11px;" in coordinate_rule


def test_chunk_overlay_annotation_boxes_do_not_overlap_the_source_grid() -> None:
    overlay = next(figure_spec for figure_spec in FIGURES if figure_spec["id"] == "chunk-overlay")
    grid = _grid_element(overlay, "chunked-source")
    root = ElementTree.fromstring(render_figure(overlay))
    grid_left, grid_top = grid["origin"]
    grid_right = grid_left + grid["columns"] * grid["cell_size"]
    grid_bottom = grid_top + grid["rows"] * grid["cell_size"]
    text_group_ids = {
        f"chunk-overlay-{element['id']}"
        for element in overlay["elements"]
        if element["kind"] == "text"
    }
    annotation_boxes = [
        child
        for group in root.iter()
        if group.attrib.get("id") in text_group_ids
        for child in group
        if child.tag.endswith("rect")
    ]

    for box in annotation_boxes:
        left = float(box.attrib["x"])
        top = float(box.attrib["y"])
        right = left + float(box.attrib["width"])
        bottom = top + float(box.attrib["height"])
        assert right <= grid_left or grid_right <= left or bottom <= grid_top or grid_bottom <= top


def test_indexing_selection_is_centered_and_phone_readable_without_horizontal_discovery() -> None:
    spec = next(figure_spec for figure_spec in FIGURES if figure_spec["id"] == "indexing-selection")
    selector = '.zi-figure[aria-labelledby~="indexing-selection-title"] {'
    figure_rule = diagram_render.STYLESHEET.split(selector, maxsplit=1)[1].split("}", maxsplit=1)[0]
    scroll_rule = diagram_render.STYLESHEET.split(
        '.zi-figure-scroll .zi-figure[aria-labelledby~="indexing-selection-title"] {',
        maxsplit=1,
    )[1].split("}", maxsplit=1)[0]

    assert spec["width"] == 360
    assert spec["width"] <= 390
    assert "max-width: 440px;" in figure_rule
    assert "margin-inline: auto;" in figure_rule
    assert f"min-width: {spec['width']}px;" in scroll_rule
    assert "#indexing-selection-result-coordinate-label" not in diagram_render.STYLESHEET


def test_prepend_coordinates_keep_global_semantics_and_old_positions() -> None:
    prepend = next(figure_spec for figure_spec in FIGURES if figure_spec["id"] == "prepend-chunk")
    nodes = {element["id"]: element for element in prepend["elements"] if element["kind"] == "node"}
    for coordinate in range(-3, 0):
        after = nodes[f"after-{coordinate}"]
        assert after["role"] == "source"
        assert after["role"] != "chunk-local"
    for coordinate in range(6):
        assert nodes[f"before-{coordinate}"]["x"] == nodes[f"after-{coordinate}"]["x"]
    assert any(
        element["kind"] == "text"
        and element["role"] == "selected"
        and element["label"] == "prepended cells"
        for element in prepend["elements"]
    )


def test_rendered_figures_have_structural_and_visible_semantics() -> None:
    document_ids: set[str] = set()
    for figure_spec in FIGURES:
        root = ElementTree.fromstring(render_figure(figure_spec))
        ids = [element.attrib["id"] for element in root.iter() if "id" in element.attrib]
        assert len(ids) == len(set(ids)), figure_spec["id"]
        assert document_ids.isdisjoint(ids), figure_spec["id"]
        document_ids.update(ids)

        labelled_by = root.attrib["aria-labelledby"].split()
        assert len(labelled_by) == 2
        assert set(labelled_by) <= set(ids)
        assert any(element.tag.endswith("title") for element in root)
        assert any(element.tag.endswith("desc") for element in root)

        role_groups = [element for element in root.iter() if "data-semantic-role" in element.attrib]
        assert role_groups, figure_spec["id"]
        for group in role_groups:
            role = group.attrib["data-semantic-role"]
            if "zi-selected-cell" in group.attrib.get("class", "").split():
                assert group.attrib["aria-label"].startswith("selected coordinate ")
            else:
                assert group.attrib["aria-label"] == role
            assert group.attrib["data-non-color-cue"] == next(
                cue for known_role, cue in ROLE_CUES.items() if known_role == role
            )

        visible_roles = {
            element.attrib["data-semantic-role"]
            for element in root.iter()
            if "zi-role-label" in element.attrib.get("class", "").split()
        }
        reader_facing_roles = {
            group.attrib["data-semantic-role"]
            for group in role_groups
            if group.attrib["data-semantic-role"] != "text"
        }
        if figure_spec.get("show_legend", True):
            assert reader_facing_roles <= visible_roles
        else:
            assert not visible_roles


def test_legend_omits_internal_text_role_without_removing_text_styling() -> None:
    for figure_spec in FIGURES:
        root = ElementTree.fromstring(render_figure(figure_spec))
        if not figure_spec.get("show_legend", True):
            assert not any(
                element.attrib.get("id") == f"{figure_spec['id']}-legend" for element in root.iter()
            )
            continue
        legend = next(
            element
            for element in root.iter()
            if element.attrib.get("id") == f"{figure_spec['id']}-legend"
        )
        assert not any(
            element.attrib.get("data-semantic-role") == "text" for element in legend.iter()
        ), figure_spec["id"]

    text_element = cast(TextSpec, {**TEXT, "role": "text"})
    root = ElementTree.fromstring(render_figure(figure("text-only", elements=(text_element,))))
    assert any(
        element.attrib.get("id") == "text-only-text"
        and element.attrib.get("class") == "zi-role-text"
        for element in root.iter()
    )
    assert not any(element.attrib.get("class") == "zi-legend" for element in root.iter())


def test_generated_stylesheet_has_theme_roles_and_responsive_layout() -> None:
    stylesheet = diagram_render.STYLESHEET
    for scheme in ("default", "slate"):
        assert f'[data-md-color-scheme="{scheme}"]' in stylesheet
        for role in get_args(SemanticRole):
            for attribute in ("fill", "stroke", "text"):
                assert (
                    f"--zi-{role}-{attribute}: {PALETTES[scheme][role][attribute]};" in stylesheet
                )
            assert f".zi-role-{role}" in stylesheet
    assert ".zi-figure" in stylesheet
    assert "max-width: 100%;" in stylesheet
    assert "@media (max-width: 600px)" in stylesheet
    assert "flex-wrap: wrap;" in stylesheet
    assert "overflow-x: visible;" in stylesheet


def test_wrapped_figures_get_their_registered_capped_minimum_width() -> None:
    stylesheet = diagram_render.STYLESHEET
    selector_prefix = '.zi-figure-scroll .zi-figure[aria-labelledby~="'

    assert stylesheet.count(selector_prefix) == len(FIGURES)
    for spec in FIGURES:
        selector = f'{selector_prefix}{spec["id"]}-title"] {{'
        rule = stylesheet.split(selector, maxsplit=1)[1].split("}", maxsplit=1)[0]
        minimum_width_match = re.search(r"min-width: (?P<width>\d+)px;", rule)
        assert minimum_width_match is not None
        assert int(minimum_width_match.group("width")) == min(spec["width"], 760)


def test_guide_figure_scroll_is_confined_without_global_minimum_width() -> None:
    stylesheet = diagram_render.STYLESHEET
    wrapper_rule = stylesheet.split(".zi-figure-scroll {", maxsplit=1)[1].split("}", maxsplit=1)[0]

    assert "max-width: 100%;" in wrapper_rule
    assert "overflow-x: auto;" in wrapper_rule
    assert "zi-lifecycle-scroll" not in stylesheet

    global_figure_rule = stylesheet.split(".zi-figure {", maxsplit=1)[1].split("}", maxsplit=1)[0]
    assert "min-width" not in global_figure_rule
    assert not re.search(r"(?:html|body|\.md-main)\s*\{[^}]*overflow-x", stylesheet)


def test_generated_assets_are_current() -> None:
    render_all(DOCS / "_static", check=True)
