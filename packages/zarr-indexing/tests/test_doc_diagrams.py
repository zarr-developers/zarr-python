from __future__ import annotations

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


def test_grid_chunk_shape_renders_chunk_boundaries() -> None:
    grid: GridSpec = {
        **GRID,
        "rows": 6,
        "columns": 8,
        "cell_size": 10,
        "origin": (5, 7),
        "chunk_shape": (3, 4),
    }
    root = ElementTree.fromstring(render_figure(figure("chunked", elements=(grid,))))
    boundaries = [
        line.attrib
        for line in root.iter()
        if line.tag.endswith("line") and line.attrib.get("class") == "zi-chunk-boundary"
    ]
    assert boundaries == [
        {"class": "zi-chunk-boundary", "x1": "45", "x2": "45", "y1": "7", "y2": "67"},
        {"class": "zi-chunk-boundary", "x1": "5", "x2": "85", "y1": "37", "y2": "37"},
    ]
    unchunked: GridSpec = {**grid, "chunk_shape": None}
    unchunked_root = ElementTree.fromstring(
        render_figure(figure("unchunked", elements=(unchunked,)))
    )
    assert not any(
        line.tag.endswith("line") and line.attrib.get("class") == "zi-chunk-boundary"
        for line in unchunked_root.iter()
    )


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
    with pytest.raises(RuntimeError) as error:
        render_all(generated_assets, check=True)
    assert str(error.value) == f"stale generated diagram: {path}"


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
        "indexing-selection": "image[1:5, 2] maps four source cells to a length-four request.",
        "coordinate-addresses": "domain [-2, 3) gives equal status to -2, -1, 0, 1, 2.",
        "prepend-chunk": "[0, 6) becomes [-3, 6) while old coordinates stay fixed.",
        "transform-mapping": "request i maps to source (i + 1, 2).",
        "compose-views": "two view maps collapse into one direct map.",
        "chunk-overlay": "the basic selection touches chunks (0, 0) and (1, 0).",
        "projection-pair": "one cell domain points to request and chunk-local spaces.",
        "orthogonal-contrast": "rows [4, 1, 1] retain order and duplication.",
    }


def _grid_element(spec: FigureSpec, element_id: str) -> GridSpec:
    for element in spec["elements"]:
        if element["id"] == element_id and element["kind"] == "grid":
            return element
    raise AssertionError(f"missing grid {element_id!r}")


def test_basic_selection_figures_share_the_canonical_source_geometry() -> None:
    figures = {figure_spec["id"]: figure_spec for figure_spec in FIGURES}
    grids = {
        "indexing-selection": _grid_element(figures["indexing-selection"], "source-grid"),
        "chunk-overlay": _grid_element(figures["chunk-overlay"], "chunked-source"),
    }
    for grid in grids.values():
        assert grid["kind"] == "grid"
        assert (grid["rows"], grid["columns"]) == (6, 8)
        assert grid["selected"] == ((1, 2), (2, 2), (3, 2), (4, 2))
    assert grids["chunk-overlay"].get("chunk_shape") == (3, 4)
    assert (
        grids["indexing-selection"]["rows"],
        grids["indexing-selection"]["columns"],
        grids["indexing-selection"]["cell_size"],
        grids["indexing-selection"]["origin"],
        grids["indexing-selection"]["selected"],
    ) == (
        grids["chunk-overlay"]["rows"],
        grids["chunk-overlay"]["columns"],
        grids["chunk-overlay"]["cell_size"],
        grids["chunk-overlay"]["origin"],
        grids["chunk-overlay"]["selected"],
    )


def test_canonical_source_svg_has_48_cells_and_internal_3_by_4_boundaries() -> None:
    figures = {figure_spec["id"]: figure_spec for figure_spec in FIGURES}
    for figure_id, grid_id in (
        ("indexing-selection", "source-grid"),
        ("chunk-overlay", "chunked-source"),
    ):
        root = ElementTree.fromstring(render_figure(figures[figure_id]))
        grid = next(
            element
            for element in root.iter()
            if element.attrib.get("id") == f"{figure_id}-{grid_id}"
        )
        cells = [
            element
            for element in grid
            if element.tag.endswith("rect")
            and "zi-grid-cell" in element.attrib.get("class", "").split()
        ]
        assert len(cells) == 48

    overlay_root = ElementTree.fromstring(render_figure(figures["chunk-overlay"]))
    overlay_grid = next(
        element
        for element in overlay_root.iter()
        if element.attrib.get("id") == "chunk-overlay-chunked-source"
    )
    boundaries = [
        element.attrib
        for element in overlay_grid
        if element.tag.endswith("line") and element.attrib.get("class") == "zi-chunk-boundary"
    ]
    assert boundaries == [
        {"class": "zi-chunk-boundary", "x1": "588", "x2": "588", "y1": "34", "y2": "286"},
        {"class": "zi-chunk-boundary", "x1": "420", "x2": "756", "y1": "160", "y2": "160"},
    ]


def test_indexing_selection_arrow_labels_do_not_obscure_selected_cells() -> None:
    spec = next(figure_spec for figure_spec in FIGURES if figure_spec["id"] == "indexing-selection")
    root = ElementTree.fromstring(render_figure(spec))
    arrow_ids = {
        f"indexing-selection-{element['id']}"
        for element in spec["elements"]
        if element["kind"] == "arrow"
    }
    arrow_label_boxes = [
        child
        for group in root.iter()
        if group.attrib.get("id") in arrow_ids
        for child in group
        if child.tag.endswith("rect")
    ]
    selected_cells = [
        element
        for element in root.iter()
        if element.tag.endswith("rect")
        and {"zi-grid-cell", "zi-role-selected"} <= set(element.attrib.get("class", "").split())
    ]
    assert arrow_label_boxes
    assert len(selected_cells) == 4

    def bounds(rectangle: ElementTree.Element) -> tuple[float, float, float, float]:
        left = float(rectangle.attrib["x"])
        top = float(rectangle.attrib["y"])
        return (
            left,
            top,
            left + float(rectangle.attrib["width"]),
            top + float(rectangle.attrib["height"]),
        )

    for label_box in arrow_label_boxes:
        label_left, label_top, label_right, label_bottom = bounds(label_box)
        for selected_cell in selected_cells:
            cell_left, cell_top, cell_right, cell_bottom = bounds(selected_cell)
            assert (
                label_right <= cell_left
                or cell_right <= label_left
                or label_bottom <= cell_top
                or cell_bottom <= label_top
            )


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
            assert group.attrib["aria-label"] == role
            assert group.attrib["data-non-color-cue"] == next(
                cue for known_role, cue in ROLE_CUES.items() if known_role == role
            )

        visible_roles = {
            element.attrib["data-semantic-role"]
            for element in root.iter()
            if "zi-role-label" in element.attrib.get("class", "").split()
        }
        assert {group.attrib["data-semantic-role"] for group in role_groups} <= visible_roles


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


def test_generated_assets_are_current() -> None:
    render_all(DOCS / "_static", check=True)
